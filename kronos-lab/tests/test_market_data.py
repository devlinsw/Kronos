from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import market_data
from market_data import (
    MarketDataError,
    interval_config,
    parse_chart_payload,
    validate_request,
)

NY = ZoneInfo("America/New_York")


def _epoch(local_time: str) -> int:
    return int(dt.datetime.fromisoformat(local_time).replace(tzinfo=NY).timestamp())


def _payload(bars: list[tuple[str, float | int, float | int, float | int, float | int, int]]) -> dict:
    return {
        "chart": {
            "error": None,
            "result": [{
                "meta": {
                    "exchangeTimezoneName": "America/New_York",
                    "symbol": "MCD",
                    "instrumentType": "EQUITY",
                    "exchangeName": "NYQ",
                    "currency": "USD",
                },
                "timestamp": [_epoch(row[0]) for row in bars],
                "indicators": {"quote": [{
                    "open": [row[1] for row in bars],
                    "high": [row[2] for row in bars],
                    "low": [row[3] for row in bars],
                    "close": [row[4] for row in bars],
                    "volume": [row[5] for row in bars],
                }]},
            }],
        }
    }


def _session_rows(day: str) -> list[tuple[str, float, float, float, float, int]]:
    rows = []
    for i, hour in enumerate((9, 10, 11, 12, 13, 14, 15)):
        minute = 30 if i == 0 or i == 6 else 30
        opening = 100.0 + i
        rows.append((f"{day}T{hour:02d}:{minute:02d}:00", opening, opening + 1, opening - 1, opening + 0.5, 10))
    return rows


def test_validate_request_normalizes_symbol_and_enforces_model_context():
    assert validate_request(" mcd ", "1H", "250") == ("MCD", "1h", 250)
    with pytest.raises(MarketDataError, match="32.*512"):
        validate_request("MCD", "1h", 10)
    with pytest.raises(MarketDataError, match="32.*512"):
        validate_request("MCD", "1h", 513)
    with pytest.raises(MarketDataError, match="ticker"):
        validate_request("../MCD", "1h", 100)
    with pytest.raises(MarketDataError, match="interval"):
        validate_request("MCD", "2h", 100)


def test_interval_config_maps_supported_intervals_and_4h_source():
    assert interval_config("1m") == {"yahoo_interval": "1m", "range": "7d"}
    assert interval_config("1h") == {"yahoo_interval": "60m", "range": "2y"}
    assert interval_config("4h") == {"yahoo_interval": "60m", "range": "2y"}
    assert interval_config("1d") == {"yahoo_interval": "1d", "range": "5y"}
    assert interval_config("1wk") == {"yahoo_interval": "1wk", "range": "10y"}


def test_parse_chart_payload_returns_last_requested_regular_session_bars():
    rows = []
    for day in ("2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"):
        rows.append((f"{day}T08:30:00", 99, 100, 98, 99, 5))
        rows.extend(_session_rows(day))
        rows.append((f"{day}T16:00:00", 106, 106, 106, 106, 0))
    result = parse_chart_payload(
        _payload(rows), "1h", 32,
        now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
    )
    assert len(result) == 32
    assert result.timestamp.is_monotonic_increasing
    assert result.timestamp.iloc[0] == pd.Timestamp("2026-09-14 12:30", tz=NY)
    assert result.timestamp.iloc[-1] == pd.Timestamp("2026-09-18 15:30", tz=NY)
    assert (result.volume > 0).all()


def test_parse_chart_payload_drops_incomplete_current_hour():
    rows = _session_rows("2026-09-18") + _session_rows("2026-09-21")
    result = parse_chart_payload(
        _payload(rows), "1h", 8,
        now=dt.datetime(2026, 9, 21, 10, 45, tzinfo=NY),
    )
    assert len(result) == 8
    assert result.timestamp.iloc[-1] == pd.Timestamp("2026-09-21 09:30", tz=NY)


def test_four_hour_bars_are_aggregated_within_session():
    result = parse_chart_payload(
        _payload(_session_rows("2026-09-18")), "4h", 2,
        now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
    )
    assert len(result) == 2
    first, second = result.iloc[0], result.iloc[1]
    assert result.timestamp.tolist() == [pd.Timestamp("2026-09-18 09:30", tz=NY), pd.Timestamp("2026-09-18 13:30", tz=NY)]
    assert (first.open, first.high, first.low, first.close, first.volume) == (100.0, 104.0, 99.0, 103.5, 40)
    assert (second.open, second.high, second.low, second.close, second.volume) == (104.0, 107.0, 103.0, 106.5, 30)


def test_daily_and_weekly_current_partial_bars_are_excluded():
    daily_rows = [
        ("2026-09-17T09:30:00", 10, 12, 9, 11, 100),
        ("2026-09-18T09:30:00", 11, 13, 10, 12, 100),
        ("2026-09-21T09:30:00", 12, 14, 11, 13, 100),
    ]
    daily = parse_chart_payload(
        _payload(daily_rows), "1d", 2,
        now=dt.datetime(2026, 9, 21, 14, 0, tzinfo=NY),
    )
    assert daily.timestamp.dt.date.tolist() == [dt.date(2026, 9, 17), dt.date(2026, 9, 18)]

    weekly_rows = [
        ("2026-09-14T00:00:00", 10, 12, 9, 11, 500),
        ("2026-09-21T00:00:00", 11, 13, 10, 12, 100),
    ]
    weekly = parse_chart_payload(
        _payload(weekly_rows), "1wk", 1,
        now=dt.datetime(2026, 9, 22, 10, 0, tzinfo=NY),
    )
    assert weekly.timestamp.iloc[0].date() == dt.date(2026, 9, 14)


def test_parse_chart_payload_errors_when_requested_history_is_unavailable():
    with pytest.raises(MarketDataError, match="Only 7 complete bars.*requested 32"):
        parse_chart_payload(
            _payload(_session_rows("2026-09-18")), "1h", 32,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_parse_chart_payload_reports_provider_error():
    with pytest.raises(MarketDataError, match="Yahoo Finance"):
        parse_chart_payload({"chart": {"error": {"description": "No data found"}, "result": None}}, "1h", 32)


def test_four_hour_bars_omit_open_current_session_bucket():
    rows = _session_rows("2026-09-18") + _session_rows("2026-09-21")
    result = parse_chart_payload(
        _payload(rows), "4h", 2,
        now=dt.datetime(2026, 9, 21, 11, 30, tzinfo=NY),
    )
    assert len(result) == 2
    assert result.timestamp.iloc[-1] == pd.Timestamp("2026-09-18 13:30", tz=NY)


def test_intraday_fetch_rejects_non_us_exchange_timezone():
    payload = _payload(_session_rows("2026-09-18"))
    payload["chart"]["result"][0]["meta"]["exchangeTimezoneName"] = "Europe/London"
    with pytest.raises(MarketDataError, match="U.S.-listed"):
        parse_chart_payload(
            payload, "1h", 1,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_four_hour_aggregator_rejects_bucket_with_missing_hour():
    rows = [row for row in _session_rows("2026-09-18") if row[0] != "2026-09-18T12:30:00"]
    with pytest.raises(MarketDataError, match="Only 1 complete bar.*requested 2"):
        parse_chart_payload(
            _payload(rows), "4h", 2,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_weekly_bar_is_complete_on_weekend():
    payload = _payload([("2026-09-21T00:00:00", 10, 12, 9, 11, 500)])
    result = parse_chart_payload(
        payload, "1wk", 1,
        now=dt.datetime(2026, 9, 26, 10, 0, tzinfo=NY),
    )
    assert result.timestamp.iloc[0] == pd.Timestamp("2026-09-21 00:00", tz=NY)


def test_early_close_hourly_final_bar_is_complete_at_real_close():
    early_close_rows = [
        (f"2026-11-27T{hour:02d}:30:00", 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10)
        for i, hour in enumerate((9, 10, 11, 12))
    ]
    result = parse_chart_payload(
        _payload(early_close_rows), "1h", 4,
        now=dt.datetime(2026, 11, 27, 13, 10, tzinfo=NY),
    )
    assert result.timestamp.iloc[-1] == pd.Timestamp("2026-11-27 12:30", tz=NY)


def test_early_close_four_hour_bucket_uses_actual_session_close():
    early_close_rows = [
        (f"2026-11-27T{hour:02d}:30:00", 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 10)
        for i, hour in enumerate((9, 10, 11, 12))
    ]
    result = parse_chart_payload(
        _payload(early_close_rows), "4h", 1,
        now=dt.datetime(2026, 11, 27, 13, 10, tzinfo=NY),
    )
    assert result.timestamp.iloc[0] == pd.Timestamp("2026-11-27 09:30", tz=NY)
    assert result.iloc[0].volume == 40


def test_early_close_daily_bar_is_complete_at_real_close():
    result = parse_chart_payload(
        _payload([("2026-11-27T09:30:00", 100, 102, 99, 101, 500)]),
        "1d", 1,
        now=dt.datetime(2026, 11, 27, 13, 10, tzinfo=NY),
    )
    assert result.timestamp.iloc[0].date() == dt.date(2026, 11, 27)


@pytest.mark.parametrize(
    ("interval", "last", "horizon", "expected"),
    [
        ("1h", "2026-11-25T15:30:00-05:00", 2,
         ["2026-11-27T09:30:00-05:00", "2026-11-27T10:30:00-05:00"]),
        ("4h", "2026-11-27T09:30:00-05:00", 2,
         ["2026-11-30T09:30:00-05:00", "2026-11-30T13:30:00-05:00"]),
        ("1d", "2026-11-25T09:30:00-05:00", 2,
         ["2026-11-27T09:30:00-05:00", "2026-11-30T09:30:00-05:00"]),
        ("1wk", "2026-11-23T00:00:00-05:00", 2,
         ["2026-11-30T00:00:00-05:00", "2026-12-07T00:00:00-05:00"]),
    ],
)
def test_future_market_timestamps_follow_exchange_sessions(interval, last, horizon, expected):
    actual = market_data.future_market_timestamps(pd.Timestamp(last), interval, horizon)
    assert [pd.Timestamp(value).isoformat() for value in actual] == expected


@pytest.mark.parametrize(
    ("instrument_type", "exchange", "timezone", "currency"),
    [
        ("EQUITY", "TOR", "America/Toronto", "CAD"),
        ("INDEX", "SNP", "America/New_York", "USD"),
    ],
)
def test_parse_chart_payload_rejects_non_us_equity_metadata(instrument_type, exchange, timezone, currency):
    payload = _payload([("2026-09-18T09:30:00", 10.0, 12.0, 9.0, 11.0, 100)])
    meta = payload["chart"]["result"][0]["meta"]
    meta.update({
        "instrumentType": instrument_type,
        "exchangeName": exchange,
        "exchangeTimezoneName": timezone,
        "currency": currency,
    })
    with pytest.raises(MarketDataError, match="U.S.-listed USD equity"):
        parse_chart_payload(
            payload, "1d", 1,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_parse_chart_payload_requires_known_us_exchange_and_returned_symbol():
    payload = _payload(_session_rows("2026-09-18"))
    meta = payload["chart"]["result"][0]["meta"]
    meta["exchangeName"] = "PNK"
    with pytest.raises(MarketDataError, match="U.S.-listed"):
        parse_chart_payload(payload, "1h", 1, expected_ticker="MCD")

    meta["exchangeName"] = "NYQ"
    meta["symbol"] = "OTHER"
    with pytest.raises(MarketDataError, match="returned symbol"):
        parse_chart_payload(payload, "1h", 1, expected_ticker="MCD")


def test_parse_chart_payload_rejects_missing_exchange_timezone():
    payload = _payload(_session_rows("2026-09-18"))
    del payload["chart"]["result"][0]["meta"]["exchangeTimezoneName"]
    with pytest.raises(MarketDataError, match="exchange metadata"):
        parse_chart_payload(payload, "1h", 1, expected_ticker="MCD")


def test_intraday_off_grid_rows_are_not_returned():
    rows = [
        ("2026-09-18T09:31:00", 100.0, 102.0, 99.0, 101.0, 10),
        ("2026-09-18T09:36:00", 101.0, 103.0, 100.0, 102.0, 10),
    ]
    with pytest.raises(MarketDataError, match="Only 0 complete bars"):
        parse_chart_payload(
            _payload(rows), "5m", 1,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_intraday_duplicate_rows_do_not_count_twice():
    row = ("2026-09-18T09:30:00", 100.0, 102.0, 99.0, 101.0, 10)
    with pytest.raises(MarketDataError, match="Only 1 complete bar"):
        parse_chart_payload(
            _payload([row, row]), "1h", 2,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_intraday_missing_grid_bar_is_reported():
    rows = [
        ("2026-09-18T09:30:00", 100.0, 102.0, 99.0, 101.0, 10),
        ("2026-09-18T11:30:00", 102.0, 104.0, 101.0, 103.0, 10),
    ]
    with pytest.raises(MarketDataError, match="missing or off-grid"):
        parse_chart_payload(
            _payload(rows), "1h", 2,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_historical_daily_weekend_bar_is_not_complete_history():
    payload = _payload([("2026-09-19T09:30:00", 100.0, 102.0, 99.0, 101.0, 10)])
    with pytest.raises(MarketDataError, match="Only 0 complete bars"):
        parse_chart_payload(
            payload, "1d", 1,
            now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
        )


def test_historical_weekly_non_monday_label_is_not_complete_history():
    payload = _payload([("2026-09-23T00:00:00", 100.0, 102.0, 99.0, 101.0, 10)])
    with pytest.raises(MarketDataError, match="Only 0 complete bars"):
        parse_chart_payload(
            payload, "1wk", 1,
            now=dt.datetime(2026, 10, 5, 9, 30, tzinfo=NY),
        )


def test_future_market_timestamps_extend_calendar_to_requested_horizon():
    actual = market_data.future_market_timestamps(
        pd.Timestamp("2040-01-03 09:30", tz=NY), "1d", 1,
    )
    assert actual == [dt.datetime(2040, 1, 4, 9, 30, tzinfo=NY)]


def test_parse_chart_payload_accepts_long_consecutive_history_without_forecast_limit():
    rows = []
    for session in pd.date_range("2026-01-05", periods=80, freq="B"):
        rows.extend(_session_rows(session.date().isoformat()))
    result = parse_chart_payload(
        _payload(rows), "1h", 240,
        now=dt.datetime(2026, 9, 21, 9, 30, tzinfo=NY),
    )
    assert len(result) == 240
