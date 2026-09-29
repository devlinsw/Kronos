from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import logging
import math
import os
import secrets
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from market_data import (
    MarketDataError,
    fetch_market_candles,
    future_market_timestamps,
    validate_request,
)

ROOT = Path(__file__).resolve().parent
KRONOS_ROOT = ROOT / "vendor" / "Kronos"
sys.path.insert(0, str(KRONOS_ROOT))

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_FLOAT32 = 3.4028234663852886e38
MAX_ROWS = 1_000_000
MAX_CONTEXT = 512
MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
TAILSCALE_IPV4_NETWORK = ipaddress.ip_network("100.64.0.0/10")
BIND_ADDRESS = os.getenv("KRONOS_TAILSCALE_IP", "127.0.0.1")
APP_USER = os.getenv("KRONOS_APP_USER", "kronos")
APP_PASSWORD = os.getenv("KRONOS_APP_PASSWORD", "")


def validate_bind_configuration(bind_address: str, password: str) -> None:
    try:
        address = ipaddress.ip_address(bind_address)
    except ValueError as exc:
        raise RuntimeError("KRONOS_TAILSCALE_IP must be a valid IPv4 address.") from exc
    if address.is_loopback:
        return
    if address.version != 4 or address not in TAILSCALE_IPV4_NETWORK:
        raise RuntimeError(
            "KRONOS_TAILSCALE_IP must be 127.0.0.1 or an IPv4 Tailscale address in "
            "100.64.0.0/10; wildcard and public-interface binds are refused."
        )
    if len(password) < 24:
        raise RuntimeError(
            "KRONOS_APP_PASSWORD must contain at least 24 characters when binding to Tailscale."
        )


validate_bind_configuration(BIND_ADDRESS, APP_PASSWORD)


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in {"/forecast", "/forecast-market"}
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except (TypeError, ValueError):
            content_length = 0
        if content_length > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total_bytes += len(chunk)
            if total_bytes > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        response = PlainTextResponse(
            "Request body exceeds the configured upload limit.",
            status_code=413,
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)

app = FastAPI(title="Kronos Lab", docs_url=None, redoc_url=None)
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Environment(
    loader=FileSystemLoader(str(ROOT / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)
_model_lock = threading.Lock()
_predictor: Any = None
logger = logging.getLogger("kronos_lab")


@app.middleware("http")
async def require_app_password(request: Request, call_next: Any) -> Any:
    if not APP_PASSWORD or request.url.path == "/health":
        return await call_next(request)
    scheme, separator, encoded = request.headers.get("authorization", "").partition(" ")
    authenticated = False
    if separator and scheme.lower() == "basic":
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            username, credential_separator, password = decoded.partition(":")
            authenticated = (
                bool(credential_separator)
                and secrets.compare_digest(username.encode("utf-8"), APP_USER.encode("utf-8"))
                and secrets.compare_digest(password.encode("utf-8"), APP_PASSWORD.encode("utf-8"))
            )
        except (binascii.Error, UnicodeDecodeError):
            authenticated = False
    if not authenticated:
        return PlainTextResponse(
            "Authentication required.",
            status_code=401,
            headers={
                "WWW-Authenticate": 'Basic realm="Kronos Lab", charset="UTF-8"',
                "Connection": "close",
            },
        )
    return await call_next(request)


def render(name: str, **context: Any) -> HTMLResponse:
    context.setdefault("horizon", 12)
    context.setdefault("ticker", "MCD")
    context.setdefault("interval", "1h")
    context.setdefault("bars", 240)
    context.setdefault("source_label", None)
    return HTMLResponse(templates.get_template(name).render(**context))


def parse_candles(payload: bytes) -> pd.DataFrame:
    if not payload:
        raise ValueError("The uploaded CSV is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the 32 MiB upload limit.")
    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    if raw.empty:
        raise ValueError("The CSV contains no candle rows.")
    if len(raw) > MAX_ROWS:
        raise ValueError(f"CSV has more than {MAX_ROWS:,} rows.")

    aliases = {
        "timestamp": {"timestamp", "datetime", "date", "time", "dt", "timestamps"},
        "open": {"open", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "c"},
        "volume": {"volume", "vol", "v"},
        "amount": {"amount", "turnover"},
    }
    source_for: dict[str, Any] = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        for canonical, names in aliases.items():
            if key in names and canonical not in source_for:
                source_for[canonical] = col
                break

    missing = [name for name in ("timestamp", "open", "high", "low", "close") if name not in source_for]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))

    result = pd.DataFrame()
    try:
        result["timestamp"] = pd.to_datetime(raw[source_for["timestamp"]], errors="coerce")
    except Exception as exc:
        raise ValueError("Timestamp column could not be parsed.") from exc
    if result["timestamp"].isna().any():
        raise ValueError("Timestamp column contains blank or invalid values.")
    try:
        for col in ("open", "high", "low", "close"):
            result[col] = pd.to_numeric(raw[source_for[col]], errors="coerce")
        if "volume" in source_for:
            result["volume"] = pd.to_numeric(raw[source_for["volume"]], errors="coerce")
        else:
            result["volume"] = 0.0
        if "amount" in source_for:
            result["amount"] = pd.to_numeric(raw[source_for["amount"]], errors="coerce")
    except Exception as exc:
        raise ValueError("OHLCV columns must contain numeric values.") from exc

    numeric_cols = ["open", "high", "low", "close", "volume"]
    if "amount" in result:
        numeric_cols.append("amount")
    if result[numeric_cols].isna().any().any():
        raise ValueError("OHLCV columns contain blank or non-numeric values.")
    for col in numeric_cols:
        if not result[col].map(math.isfinite).all():
            raise ValueError(f"Column {col} contains a non-finite value.")
    if (result[numeric_cols].abs() > MAX_FLOAT32).any().any():
        raise ValueError("OHLCV values exceed the model's float32 input range.")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (result["volume"] < 0).any() or ("amount" in result and (result["amount"] < 0).any()):
        raise ValueError("Volume and amount cannot be negative.")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("At least one row has a high below its open, close, or low.")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("At least one row has a low above its open, close, or high.")

    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("Timestamp column contains duplicate bars.")
    if len(result) < 32:
        raise ValueError("Upload at least 32 valid candle rows.")
    return result


def infer_future_timestamps(timestamps: pd.Series, horizon: int) -> pd.DatetimeIndex:
    deltas = timestamps.diff().dropna()
    seconds = deltas.dt.total_seconds()
    positive = seconds[seconds > 0]
    if positive.empty:
        raise ValueError("Could not infer a positive bar interval from timestamps.")
    step = pd.to_timedelta(float(positive.tail(128).median()), unit="s")
    if step <= pd.Timedelta(0):
        raise ValueError("Could not infer a positive bar interval from timestamps.")
    last = timestamps.iloc[-1]
    return pd.DatetimeIndex([last + step * i for i in range(1, horizon + 1)])


def get_predictor() -> Any:
    global _predictor
    if _predictor is not None:
        return _predictor
    with _model_lock:
        if _predictor is None:
            import torch
            from model import Kronos, KronosPredictor, KronosTokenizer

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
            model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            tokenizer.eval()
            model.eval()
            _predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    return _predictor


def build_chart(history: list[float], forecast: list[float]) -> str:
    values = history + forecast
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    width, height, pad = 920.0, 240.0, 18.0
    xstep = (width - 2 * pad) / max(1, len(values) - 1)

    def point(index: int, value: float) -> str:
        x = pad + index * xstep
        y = height - pad - ((value - lo) / span) * (height - 2 * pad)
        return f"{x:.1f},{y:.1f}"

    actual = " ".join(point(i, value) for i, value in enumerate(history))
    predicted = " ".join(point(len(history) - 1 + i, value) for i, value in enumerate(forecast))
    split_x = pad + (len(history) - 1) * xstep
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Close-price history and Kronos forecast">'
        f'<line x1="{split_x:.1f}" y1="8" x2="{split_x:.1f}" y2="{height-8:.1f}" class="chart-divider" />'
        f'<polyline points="{actual}" class="chart-history" />'
        f'<polyline points="{predicted}" class="chart-forecast" />'
        f"</svg>"
    )


def render_error(
    message: str,
    horizon: int = 12,
    ticker: str = "MCD",
    interval: str = "1h",
    bars: int | str = 240,
) -> HTMLResponse:
    return render(
        "index.html",
        model_loaded=_predictor is not None,
        forecast=None,
        error=message,
        chart_svg=None,
        data_rows=0,
        horizon=horizon,
        ticker=ticker,
        interval=interval,
        bars=bars,
        source_label=None,
    )


def prepare_model_input(candles: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError("Model input is missing columns: " + ", ".join(missing))

    model_input = candles.tail(MAX_CONTEXT).reset_index(drop=True).copy()
    if "volume" not in model_input.columns:
        model_input["volume"] = 0.0
    columns = required + ["volume"]
    values = model_input[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("Model input contains a blank or non-numeric OHLCV value.")
    for column in columns:
        if not values[column].map(math.isfinite).all():
            raise ValueError(f"Model input column {column} contains a non-finite value.")
        if (values[column].abs() > MAX_FLOAT32).any():
            raise ValueError(f"Model input column {column} exceeds the float32 range.")
    if (values[required] <= 0).any().any():
        raise ValueError("Model input OHLC prices must be positive.")
    if (values["volume"] < 0).any():
        raise ValueError("Model input volume cannot be negative.")
    if (values["high"] < values[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Model input has a high below another OHLC value.")
    if (values["low"] > values[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Model input has a low above another OHLC value.")

    if "amount" in model_input.columns:
        amount = pd.to_numeric(model_input["amount"], errors="coerce")
    else:
        amount = values["volume"] * values[required].mean(axis=1)
    if amount.isna().any() or not amount.map(math.isfinite).all():
        raise ValueError("Model input amount contains a non-finite value.")
    if (amount.abs() > MAX_FLOAT32).any():
        raise ValueError("Model input amount exceeds the float32 range.")
    if (amount < 0).any():
        raise ValueError("Model input amount cannot be negative.")

    model_input[columns] = values
    model_input["amount"] = amount
    return model_input, columns + ["amount"]


def validate_forecast_output(
    prediction: pd.DataFrame,
    horizon: int,
    expected_timestamps: pd.Index | list[pd.Timestamp],
) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in prediction.columns]
    if missing:
        raise ValueError("Forecast output is missing columns: " + ", ".join(missing))
    if len(prediction) != horizon:
        raise ValueError(f"Model returned {len(prediction)} rows; expected {horizon} forecast rows.")
    try:
        expected_index = pd.DatetimeIndex(expected_timestamps)
        actual_index = pd.DatetimeIndex(prediction.index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model returned timestamps that do not match the requested forecast.") from exc
    if not actual_index.equals(expected_index):
        raise ValueError("Model returned timestamps that do not match the requested forecast.")
    checked_columns = required + (["amount"] if "amount" in prediction.columns else [])
    values = prediction[checked_columns].apply(pd.to_numeric, errors="coerce")
    if not all(math.isfinite(float(value)) for value in values.to_numpy().flat):
        raise ValueError("Model returned a non-finite forecast value.")
    if (values.abs() > MAX_FLOAT32).any().any():
        raise ValueError("Model returned a forecast outside the float32 range.")
    ohlc = values[["open", "high", "low", "close"]]
    if (ohlc <= 0).any().any():
        raise ValueError("Model returned a non-positive forecast price.")
    if (values["volume"] < 0).any():
        raise ValueError("Model returned a negative forecast volume.")
    if "amount" in values and (values["amount"] < 0).any():
        raise ValueError("Model returned a negative forecast amount.")
    if (values["high"] < ohlc[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Model returned a high below another OHLC value.")
    if (values["low"] > ohlc[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Model returned a low above another OHLC value.")
    return prediction


def run_forecast(
    candles: pd.DataFrame,
    horizon: int,
    source_label: str,
    market_interval: str | None = None,
) -> HTMLResponse:
    if market_interval is None:
        future_ts = infer_future_timestamps(candles["timestamp"], horizon)
    else:
        future_ts = future_market_timestamps(candles["timestamp"].iloc[-1], market_interval, horizon)
    model_input, model_columns = prepare_model_input(candles)
    prediction = get_predictor().predict(
        df=model_input[model_columns],
        x_timestamp=model_input["timestamp"],
        y_timestamp=pd.Series(future_ts),
        pred_len=horizon,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    prediction = validate_forecast_output(prediction, horizon, future_ts)
    records = []
    for timestamp, row in prediction.iterrows():
        records.append({
            "timestamp": pd.Timestamp(timestamp).isoformat(sep=" "),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    history = [float(value) for value in model_input["close"].tail(120)]
    predicted = [record["close"] for record in records]
    return render(
        "index.html",
        model_loaded=True,
        forecast=records,
        error=None,
        chart_svg=build_chart(history, predicted),
        data_rows=len(candles),
        horizon=horizon,
        first_timestamp=records[0]["timestamp"],
        last_timestamp=records[-1]["timestamp"],
        model_id=MODEL_ID,
        source_label=source_label,
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return render(
        "index.html",
        model_loaded=_predictor is not None,
        forecast=None,
        error=None,
        chart_svg=None,
        data_rows=0,
        horizon=12,
        ticker="MCD",
        interval="1h",
        bars=240,
        source_label=None,
    )


@app.post("/forecast", response_class=HTMLResponse)
def forecast(file: UploadFile = File(...), horizon: int = Form(12)) -> HTMLResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return render_error("Upload a .csv file.", horizon=horizon)
    if not 1 <= horizon <= 128:
        return render_error("Forecast horizon must be between 1 and 128 bars.", horizon=horizon)
    try:
        payload = file.file.read(MAX_UPLOAD_BYTES + 1)
        candles = parse_candles(payload)
        return run_forecast(candles, horizon, "CSV upload")
    except ValueError as exc:
        return render_error(str(exc), horizon=horizon)
    except Exception as exc:
        logger.exception("CSV forecast failed")
        return render_error(
            f"Forecast failed ({type(exc).__name__}). Check the service logs; no order was sent.",
            horizon=horizon,
        )


@app.post("/forecast-market", response_class=HTMLResponse)
def forecast_market(
    ticker: str = Form("MCD"),
    interval: str = Form("1h"),
    bars: str = Form("240"),
    horizon: int = Form(12),
) -> HTMLResponse:
    normalized_ticker, normalized_interval, count = ticker, interval, bars
    try:
        normalized_ticker, normalized_interval, count = validate_request(ticker, interval, bars)
        if not 1 <= horizon <= 128:
            raise MarketDataError("Forecast horizon must be between 1 and 128 bars.")
        candles = fetch_market_candles(normalized_ticker, normalized_interval, count)
        if len(candles) != count:
            raise MarketDataError(f"Provider returned {len(candles)} bars; expected exactly {count}.")
        response = run_forecast(
            candles,
            horizon,
            f"Yahoo Finance · {normalized_ticker} · {normalized_interval}",
            market_interval=normalized_interval,
        )
        return HTMLResponse(response.body, headers={
            "X-Kronos-Ticker": normalized_ticker,
            "X-Kronos-Interval": normalized_interval,
            "X-Kronos-Bars": str(count),
        })
    except ValueError as exc:
        return render_error(
            str(exc), horizon=horizon, ticker=normalized_ticker,
            interval=normalized_interval, bars=count,
        )
    except Exception as exc:
        logger.exception("Market-data forecast failed for %s %s", normalized_ticker, normalized_interval)
        return render_error(
            f"Market forecast failed ({type(exc).__name__}). Check the service logs; no order was sent.",
            horizon=horizon, ticker=normalized_ticker,
            interval=normalized_interval, bars=count,
        )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "kronos-lab",
        "model_loaded": _predictor is not None,
        "model": MODEL_ID,
        "forecast_mode": "greedy_single_path",
        "decision_backend": "disabled",
    })
