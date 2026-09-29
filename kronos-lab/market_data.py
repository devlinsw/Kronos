from __future__ import annotations

import datetime as dt
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import exchange_calendars as xcals
import pandas as pd

MIN_BARS = 32
MAX_BARS = 512
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MARKET_TIMEZONE = ZoneInfo("America/New_York")
SESSION_FINALIZE_DELAY = dt.timedelta(minutes=5)
# Yahoo's public chart endpoint is a prototype source, not a contractual feed.
# Four-hour candles are built from complete hourly bars, aligned to the XNYS open.
_INTERVALS = {
    "1m": {"yahoo_interval": "1m", "range": "7d"},
    "5m": {"yahoo_interval": "5m", "range": "60d"},
    "15m": {"yahoo_interval": "15m", "range": "60d"},
    "30m": {"yahoo_interval": "30m", "range": "60d"},
    "1h": {"yahoo_interval": "60m", "range": "2y"},
    "4h": {"yahoo_interval": "60m", "range": "2y"},
    "1d": {"yahoo_interval": "1d", "range": "5y"},
    "1wk": {"yahoo_interval": "1wk", "range": "10y"},
}
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9.^=-]{1,15}\Z")
_INTRADAY_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
_US_EQUITY_EXCHANGES = {"NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS"}


class MarketDataError(ValueError):
    """A safe, user-displayable market-data validation or provider error."""


def validate_request(ticker: str, interval: str, bars: int | str) -> tuple[str, str, int]:
    symbol = str(ticker).strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise MarketDataError("Enter a valid ticker symbol (letters, numbers, dot, hyphen, or ^).")
    timeframe = str(interval).strip().lower()
    if timeframe not in _INTERVALS:
        raise MarketDataError("Choose a supported candle interval.")
    try:
        count = int(bars)
    except (TypeError, ValueError) as exc:
        raise MarketDataError("Candle count must be a whole number from 32 to 512.") from exc
    if str(bars).strip() not in {str(count), f"+{count}"} or not MIN_BARS <= count <= MAX_BARS:
        raise MarketDataError(f"Candle count must be between {MIN_BARS} and {MAX_BARS}.")
    return symbol, timeframe, count


def interval_config(interval: str) -> dict[str, str]:
    try:
        return dict(_INTERVALS[interval])
    except KeyError as exc:
        raise MarketDataError("Choose a supported candle interval.") from exc


def _now_in_timezone(now: dt.datetime | None, timezone: ZoneInfo) -> dt.datetime:
    if now is None:
        return dt.datetime.now(timezone)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone)
    return now.astimezone(timezone)


def _is_valid_bar(values: tuple[float, float, float, float, float]) -> bool:
    opening, high, low, close, volume = values
    return (
        all(math.isfinite(value) for value in values)
        and min(opening, high, low, close) > 0
        and volume >= 0
        and high >= max(opening, close, low)
        and low <= min(opening, close, high)
    )


@lru_cache(maxsize=8)
def _calendar(end_year: int = 2035):
    return xcals.get_calendar("XNYS", start="2000-01-01", end=f"{max(2035, end_year)}-12-31")


@lru_cache(maxsize=8192)
def _session_times(
    session_date: dt.date,
    calendar_end_year: int | None = None,
) -> tuple[dt.datetime, dt.datetime] | None:
    label = pd.Timestamp(session_date)
    try:
        calendar = _calendar(max(2035, calendar_end_year or session_date.year))
        if not calendar.is_session(label):
            return None
        opening, closing = calendar.session_open_close(label)
    except (IndexError, KeyError, ValueError):
        return None
    return (
        opening.to_pydatetime().astimezone(MARKET_TIMEZONE),
        closing.to_pydatetime().astimezone(MARKET_TIMEZONE),
    )


def _week_last_session_close(week_start: dt.date) -> dt.datetime | None:
    week_end = week_start + dt.timedelta(days=6)
    calendar_end_year = max(2035, week_end.year)
    calendar = _calendar(calendar_end_year)
    sessions = calendar.sessions_in_range(pd.Timestamp(week_start), pd.Timestamp(week_end))
    if sessions.empty:
        return None
    times = _session_times(sessions[-1].date(), calendar_end_year)
    return times[1] if times else None


def _in_regular_session(timestamp: dt.datetime) -> bool:
    times = _session_times(timestamp.date())
    return bool(times and times[0] <= timestamp < times[1])


def _on_intraday_grid(timestamp: dt.datetime, interval: str) -> bool:
    times = _session_times(timestamp.date())
    if not times or not times[0] <= timestamp < times[1]:
        return False
    minutes = _INTRADAY_MINUTES[interval]
    seconds_since_open = (timestamp - times[0]).total_seconds()
    return seconds_since_open >= 0 and seconds_since_open % (minutes * 60) == 0


def _four_hour_bucket(timestamp: dt.datetime) -> tuple[dt.datetime, dt.datetime] | None:
    times = _session_times(timestamp.date())
    if not times or not times[0] <= timestamp < times[1]:
        return None
    opening, closing = times
    bucket_index = int((timestamp - opening).total_seconds() // (4 * 60 * 60))
    bucket_start = opening + dt.timedelta(hours=4 * bucket_index)
    return bucket_start, min(bucket_start + dt.timedelta(hours=4), closing)


def _intraday_complete(timestamp: dt.datetime, interval: str, now: dt.datetime) -> bool:
    times = _session_times(timestamp.date())
    if not times or not times[0] <= timestamp < times[1]:
        return False
    if timestamp.date() < now.date():
        return True
    if timestamp.date() > now.date():
        return False
    if interval == "4h":
        bucket = _four_hour_bucket(timestamp)
        if not bucket:
            return False
        bucket_end = bucket[1]
    else:
        bucket_end = min(
            timestamp + dt.timedelta(minutes=_INTRADAY_MINUTES[interval]),
            times[1],
        )
    return bucket_end <= now


def _daily_complete(timestamp: dt.datetime, now: dt.datetime) -> bool:
    if timestamp.date() < now.date():
        return True
    if timestamp.date() > now.date():
        return False
    times = _session_times(timestamp.date())
    return bool(times and now >= times[1] + SESSION_FINALIZE_DELAY)


def _weekly_complete(timestamp: dt.datetime, now: dt.datetime) -> bool:
    week_start = timestamp.date() - dt.timedelta(days=timestamp.weekday())
    current_week_start = now.date() - dt.timedelta(days=now.weekday())
    if week_start < current_week_start:
        return True
    if week_start > current_week_start:
        return False
    last_close = _week_last_session_close(week_start)
    return bool(last_close and now >= last_close + SESSION_FINALIZE_DELAY)


def _expected_hour_starts(bucket_start: dt.datetime, bucket_end: dt.datetime) -> list[dt.datetime]:
    count = math.ceil((bucket_end - bucket_start).total_seconds() / 3600)
    return [bucket_start + dt.timedelta(hours=index) for index in range(count)]


def _aggregate_four_hour(
    rows: list[tuple[dt.datetime, float, float, float, float, float]],
    now: dt.datetime,
) -> list[tuple[dt.datetime, float, float, float, float, float]]:
    groups: dict[tuple[dt.date, dt.datetime], list[tuple[dt.datetime, float, float, float, float, float]]] = {}
    for row in rows:
        timestamp = row[0]
        bucket = _four_hour_bucket(timestamp)
        if not bucket:
            continue
        bucket_start, bucket_end = bucket
        if timestamp.date() == now.date() and bucket_end > now:
            continue
        groups.setdefault((timestamp.date(), bucket_start), []).append(row)

    aggregated = []
    for (session_date, bucket_start), bucket in sorted(groups.items()):
        times = _session_times(session_date)
        if not times:
            continue
        bucket_end = min(bucket_start + dt.timedelta(hours=4), times[1])
        expected = _expected_hour_starts(bucket_start, bucket_end)
        by_timestamp = {row[0]: row for row in bucket}
        if sorted(by_timestamp) != expected:
            continue
        ordered = [by_timestamp[timestamp] for timestamp in expected]
        aggregated.append((
            bucket_start,
            ordered[0][1],
            max(row[2] for row in ordered),
            min(row[3] for row in ordered),
            ordered[-1][4],
            sum(row[5] for row in ordered),
        ))
    return aggregated


def _next_market_timestamps(
    last_timestamp: dt.datetime | pd.Timestamp,
    interval: str,
    count: int,
) -> list[dt.datetime]:
    if interval not in _INTERVALS:
        raise MarketDataError("Choose a supported candle interval.")
    if count < 1:
        return []
    last = pd.Timestamp(last_timestamp)
    last = last.tz_localize(MARKET_TIMEZONE) if last.tz is None else last.tz_convert(MARKET_TIMEZONE)
    last_dt = last.to_pydatetime()

    if interval == "1wk":
        first_week = last_dt.date() - dt.timedelta(days=last_dt.weekday()) + dt.timedelta(days=7)
        return [
            dt.datetime.combine(first_week + dt.timedelta(days=7 * index), dt.time(), tzinfo=MARKET_TIMEZONE)
            for index in range(count)
        ]

    sessions_end = last_dt.date() + dt.timedelta(days=max(365, count * 10))
    calendar_end_year = max(2035, sessions_end.year)
    calendar = _calendar(calendar_end_year)
    sessions = calendar.sessions_in_range(pd.Timestamp(last_dt.date()), pd.Timestamp(sessions_end))
    result: list[dt.datetime] = []
    if interval == "4h":
        step = dt.timedelta(hours=4)
    elif interval == "1d":
        step = dt.timedelta(0)
    else:
        step = dt.timedelta(minutes=_INTRADAY_MINUTES[interval])

    for label in sessions:
        times = _session_times(label.date(), calendar_end_year)
        if not times:
            continue
        opening, closing = times
        if interval == "1d":
            candidates = [opening]
        else:
            candidates = []
            candidate = opening
            while candidate < closing:
                candidates.append(candidate)
                candidate += step
        for candidate in candidates:
            if candidate <= last_dt:
                continue
            result.append(candidate)
            if len(result) == count:
                return result
    raise MarketDataError("Could not create future timestamps from the U.S. exchange calendar.")


def future_market_timestamps(
    last_timestamp: dt.datetime | pd.Timestamp,
    interval: str,
    horizon: int,
) -> list[dt.datetime]:
    if not 1 <= horizon <= 128:
        raise MarketDataError("Forecast horizon must be between 1 and 128 bars.")
    return _next_market_timestamps(last_timestamp, interval, horizon)


def parse_chart_payload(
    payload: dict,
    interval: str,
    bars: int,
    now: dt.datetime | None = None,
    expected_ticker: str | None = None,
) -> pd.DataFrame:
    chart = payload.get("chart", {})
    if chart.get("error"):
        detail = chart["error"].get("description") or chart["error"].get("code") or "provider error"
        raise MarketDataError(f"Yahoo Finance could not fetch this request: {detail}")
    results = chart.get("result") or []
    if not results:
        raise MarketDataError("Yahoo Finance returned no history for this ticker and interval.")
    result = results[0]
    metadata = result.get("meta", {})
    try:
        timezone_name = metadata.get("exchangeTimezoneName")
        if not timezone_name:
            raise MarketDataError("Yahoo Finance returned incomplete exchange metadata.")
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise MarketDataError("Yahoo Finance returned an unknown exchange timezone.") from exc
    returned_symbol = metadata.get("symbol")
    if not returned_symbol:
        raise MarketDataError("Yahoo Finance returned incomplete exchange metadata.")
    if expected_ticker and str(returned_symbol).upper() != expected_ticker.upper():
        raise MarketDataError(
            f"Yahoo Finance returned symbol {returned_symbol} for requested {expected_ticker}."
        )
    if (
        metadata.get("instrumentType") != "EQUITY"
        or metadata.get("currency") != "USD"
        or metadata.get("exchangeName") not in _US_EQUITY_EXCHANGES
        or timezone.key != "America/New_York"
    ):
        raise MarketDataError("Yahoo metadata does not identify this as a U.S.-listed USD equity.")
    current = _now_in_timezone(now, timezone)
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    fields = [quotes.get(name) or [] for name in ("open", "high", "low", "close", "volume")]
    raw_rows: list[tuple[dt.datetime, float, float, float, float, float]] = []

    for index, epoch in enumerate(timestamps):
        try:
            timestamp = dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc).astimezone(timezone)
            values = tuple(float(field[index]) for field in fields)
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
        if not _is_valid_bar(values):
            continue
        if interval in _INTRADAY_MINUTES or interval == "4h":
            source_interval = "1h" if interval == "4h" else interval
            if not _on_intraday_grid(timestamp, source_interval):
                continue
            if not _intraday_complete(timestamp, source_interval, current):
                continue
        elif interval == "1d":
            session_times = _session_times(timestamp.date())
            if not session_times or timestamp != session_times[0]:
                continue
            if not _daily_complete(timestamp, current):
                continue
        elif interval == "1wk":
            week_start = timestamp.date() - dt.timedelta(days=timestamp.weekday())
            week_label = dt.datetime.combine(week_start, dt.time(), tzinfo=timezone)
            if timestamp != week_label or _week_last_session_close(week_start) is None:
                continue
            if not _weekly_complete(timestamp, current):
                continue
        raw_rows.append((timestamp, *values))

    unique_rows: dict[dt.datetime, tuple[dt.datetime, float, float, float, float, float]] = {}
    for row in raw_rows:
        previous = unique_rows.get(row[0])
        if previous is not None and previous[1:] != row[1:]:
            raise MarketDataError("Yahoo Finance returned conflicting duplicate candle timestamps.")
        unique_rows[row[0]] = row
    raw_rows = list(unique_rows.values())
    if interval == "4h":
        raw_rows = _aggregate_four_hour(raw_rows, current)
    raw_rows.sort(key=lambda row: row[0])
    selected = raw_rows[-bars:]
    if len(selected) < bars:
        noun, verb = ("bar", "is") if len(selected) == 1 else ("bars", "are")
        raise MarketDataError(
            f"Only {len(selected)} complete {noun} {verb} available for this ticker/interval; requested {bars}. "
            "Try a smaller count or a different interval."
        )
    if bars > 1:
        expected = _next_market_timestamps(selected[0][0], interval, bars - 1)
        actual = [row[0] for row in selected[1:]]
        if actual != expected:
            raise MarketDataError(
                "Yahoo Finance history contains missing or off-grid candles; try fewer bars or another interval."
            )
    return pd.DataFrame(selected, columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_market_candles(ticker: str, interval: str, bars: int | str) -> pd.DataFrame:
    symbol, timeframe, count = validate_request(ticker, interval, bars)
    config = interval_config(timeframe)
    encoded_symbol = urllib.parse.quote(symbol, safe=".^=-")
    query = urllib.parse.urlencode({
        "range": config["range"],
        "interval": config["yahoo_interval"],
        "includePrePost": "false",
        "events": "div,splits",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?{query}"
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; KronosLab/0.1)",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise MarketDataError(f"Yahoo Finance request failed (HTTP {exc.code}). Try again later.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MarketDataError("Could not reach Yahoo Finance. Check network access and try again.") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise MarketDataError("Yahoo Finance returned too much data. Request fewer bars or a shorter timeframe.")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MarketDataError("Yahoo Finance returned an unreadable response. Try again later.") from exc
    return parse_chart_payload(payload, timeframe, count, expected_ticker=symbol)
