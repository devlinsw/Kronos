import asyncio
import inspect
import os
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
from market_data import MarketDataError
from app import infer_future_timestamps, parse_candles


def test_parse_candles_normalizes_columns_sorts_and_fills_volume():
    csv = (
        "Date,Open,High,Low,Close\n"
        "2025-01-02 09:32:00,101,102,100,101.5\n"
        "2025-01-02 09:31:00,100,101,99,100.5\n"
        "2025-01-02 09:33:00,101.5,103,101,102.5\n"
    )
    # Parser requires 32 bars so repeat valid minute rows with unique timestamps.
    rows = ["Date,Open,High,Low,Close"]
    start = pd.Timestamp("2025-01-02 09:31:00")
    for i in range(32):
        price = 100 + i / 10
        rows.append(f"{start + pd.Timedelta(minutes=i)},{price},{price+1},{price-1},{price+0.5}")
    result = parse_candles("\n".join(rows).encode())
    assert len(result) == 32
    assert list(result.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert result["timestamp"].is_monotonic_increasing
    assert (result["volume"] == 0).all()


def test_parse_candles_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        parse_candles(b"timestamp,open,close\n2025-01-01,1,1\n")


def test_parse_candles_rejects_inconsistent_ohlc():
    rows = ["timestamp,open,high,low,close"]
    start = pd.Timestamp("2025-01-02 09:31:00")
    for i in range(32):
        high = 100.0 if i == 0 else 102.0
        rows.append(f"{start + pd.Timedelta(minutes=i)},101,{high},99,101.5")
    with pytest.raises(ValueError, match="high below"):
        parse_candles("\n".join(rows).encode())


def test_infer_future_timestamps_uses_median_interval():
    timestamps = pd.Series(pd.date_range("2025-01-02 09:30", periods=40, freq="5min"))
    future = infer_future_timestamps(timestamps, 3)
    assert list(future) == list(pd.date_range("2025-01-02 12:50", periods=3, freq="5min"))


def test_home_exposes_market_data_controls():
    response = TestClient(app_module.app).get("/")
    assert response.status_code == 200
    assert 'action="/forecast-market"' in response.text
    assert 'name="ticker"' in response.text
    assert 'name="interval"' in response.text
    assert 'name="bars"' in response.text


def test_market_fetch_route_feeds_provider_candles_to_predictor(monkeypatch):
    index = pd.DatetimeIndex([
        pd.Timestamp(f"2026-03-{day:02d} {hour:02d}:30", tz="America/New_York")
        for day in (6, 9, 10, 11, 12)
        for hour in (9, 10, 11, 12, 13, 14, 15)
    ][:32])
    close = [100.0 + i / 10 for i in range(32)]
    candles = pd.DataFrame({
        "timestamp": index,
        "open": close,
        "high": [value + 1 for value in close],
        "low": [value - 1 for value in close],
        "close": close,
        "volume": [1000] * 32,
    })
    monkeypatch.setattr(app_module, "fetch_market_candles", lambda ticker, interval, bars: candles)
    future = [
        pd.Timestamp("2026-03-12 13:30", tz="America/New_York"),
        pd.Timestamp("2026-03-12 14:30", tz="America/New_York"),
    ]
    monkeypatch.setattr(
        app_module, "future_market_timestamps",
        lambda last_timestamp, interval, horizon: future,
        raising=False,
    )
    seen = {}

    class FakePredictor:
        def predict(self, **kwargs):
            future = kwargs["y_timestamp"]
            seen["y_timestamp"] = future.tolist()
            seen["x_timestamp"] = kwargs["x_timestamp"]
            return pd.DataFrame({
                "open": [103.0, 104.0], "high": [104.0, 105.0],
                "low": [102.0, 103.0], "close": [103.5, 104.5],
                "volume": [900.0, 950.0],
            }, index=future)

    monkeypatch.setattr(app_module, "get_predictor", lambda: FakePredictor())
    response = TestClient(app_module.app).post("/forecast-market", data={
        "ticker": "mcd", "interval": "1h", "bars": "32", "horizon": "2",
    })
    assert response.status_code == 200
    assert "32 input bars" in response.text
    assert "MCD · 1h" in response.text
    assert response.text.count('class="close-cell"') == 2
    assert seen["y_timestamp"] == future
    assert str(seen["x_timestamp"].dt.tz) == "America/New_York"


def test_market_fetch_route_displays_provider_errors(monkeypatch):
    def fail_fetch(ticker, interval, bars):
        raise MarketDataError("Yahoo Finance returned no history for this ticker and interval.")

    monkeypatch.setattr(app_module, "fetch_market_candles", fail_fetch)
    response = TestClient(app_module.app).post("/forecast-market", data={
        "ticker": "NOTAREALTICKER", "interval": "1h", "bars": "32", "horizon": "12",
    })
    assert response.status_code == 200
    assert "Yahoo Finance returned no history" in response.text


def test_market_fetch_route_displays_invalid_candle_count(monkeypatch):
    monkeypatch.setattr(
        app_module, "fetch_market_candles",
        lambda *args: pytest.fail("provider must not run for an invalid count"),
    )
    response = TestClient(app_module.app).post("/forecast-market", data={
        "ticker": "MCD", "interval": "1h", "bars": "many", "horizon": "12",
    })
    assert response.status_code == 200
    assert "Candle count must be a whole number" in response.text


def test_parse_candles_rejects_values_outside_model_float32_range():
    rows = ["timestamp,open,high,low,close"]
    start = pd.Timestamp("2025-01-02 09:31:00")
    for i in range(32):
        open_price = 1e39 if i == 0 else 100.0 + i
        high = open_price * 1.01 if i == 0 else open_price + 1
        low = open_price * 0.99 if i == 0 else open_price - 1
        rows.append(f"{start + pd.Timedelta(minutes=i)},{open_price},{high},{low},{open_price}")
    with pytest.raises(ValueError, match="float32"):
        parse_candles("\n".join(rows).encode())


def test_run_forecast_rejects_nonfinite_model_output(monkeypatch):
    timestamps = pd.date_range("2025-01-02 09:31", periods=32, freq="min")
    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 32,
        "high": [101.0] * 32,
        "low": [99.0] * 32,
        "close": [100.0] * 32,
        "volume": [10.0] * 32,
    })

    class InvalidPredictor:
        def predict(self, **kwargs):
            return pd.DataFrame({
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [float("nan")], "volume": [10.0],
            }, index=kwargs["y_timestamp"])

    monkeypatch.setattr(app_module, "get_predictor", lambda: InvalidPredictor())
    with pytest.raises(ValueError, match="non-finite forecast"):
        app_module.run_forecast(candles, 1, "test")


def test_run_forecast_rejects_inconsistent_model_ohlc(monkeypatch):
    timestamps = pd.date_range("2025-01-02 09:31", periods=32, freq="min")
    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 32,
        "high": [101.0] * 32,
        "low": [99.0] * 32,
        "close": [100.0] * 32,
        "volume": [10.0] * 32,
    })

    class InvalidPredictor:
        def predict(self, **kwargs):
            return pd.DataFrame({
                "open": [100.0], "high": [98.0], "low": [97.0],
                "close": [98.0], "volume": [10.0],
            }, index=kwargs["y_timestamp"])

    monkeypatch.setattr(app_module, "get_predictor", lambda: InvalidPredictor())
    with pytest.raises(ValueError, match="high below"):
        app_module.run_forecast(candles, 1, "test")


def test_run_forecast_rejects_market_values_outside_float32_range(monkeypatch):
    timestamps = pd.date_range("2025-01-02 09:31", periods=32, freq="min")
    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": [1e39] * 32,
        "high": [1.1e39] * 32,
        "low": [0.9e39] * 32,
        "close": [1e39] * 32,
        "volume": [10.0] * 32,
    })
    monkeypatch.setattr(app_module, "get_predictor", lambda: pytest.fail("invalid input reached model"))
    with pytest.raises(ValueError, match="float32"):
        app_module.run_forecast(candles, 1, "market test")


def test_run_forecast_rejects_derived_amount_overflow(monkeypatch):
    timestamps = pd.date_range("2025-01-02 09:31", periods=32, freq="min")
    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 32,
        "high": [101.0] * 32,
        "low": [99.0] * 32,
        "close": [100.0] * 32,
        "volume": [1e37] * 32,
    })
    monkeypatch.setattr(app_module, "get_predictor", lambda: pytest.fail("derived overflow reached model"))
    with pytest.raises(ValueError, match="amount.*float32"):
        app_module.run_forecast(candles, 1, "derived-amount test")


def test_run_forecast_rejects_wrong_forecast_timestamps(monkeypatch):
    timestamps = pd.date_range("2025-01-02 09:31", periods=32, freq="min")
    candles = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100.0] * 32,
        "high": [101.0] * 32,
        "low": [99.0] * 32,
        "close": [100.0] * 32,
        "volume": [10.0] * 32,
    })

    class WrongTimestampPredictor:
        def predict(self, **kwargs):
            wrong_timestamp = pd.DatetimeIndex(kwargs["y_timestamp"]) + pd.Timedelta(days=1)
            return pd.DataFrame({
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.0], "volume": [10.0],
            }, index=wrong_timestamp)

    monkeypatch.setattr(app_module, "get_predictor", lambda: WrongTimestampPredictor())
    with pytest.raises(ValueError, match="timestamp.*requested"):
        app_module.run_forecast(candles, 1, "timestamp test")


def test_csv_forecast_handler_is_sync_for_threadpool_execution():
    assert not inspect.iscoroutinefunction(app_module.forecast)


def test_csv_upload_reaches_predictor_through_request_limit_middleware(monkeypatch):
    rows = ["timestamp,open,high,low,close,volume"]
    start = pd.Timestamp("2025-01-02 09:31:00")
    for i in range(32):
        price = 100.0 + i / 10
        rows.append(f"{start + pd.Timedelta(minutes=i)},{price},{price+1},{price-1},{price+0.5},1000")
    csv_bytes = "\n".join(rows).encode()

    class FakePredictor:
        def predict(self, **kwargs):
            return pd.DataFrame({
                "open": [104.0], "high": [105.0], "low": [103.0],
                "close": [104.5], "volume": [1200.0],
            }, index=kwargs["y_timestamp"])

    monkeypatch.setattr(app_module, "get_predictor", lambda: FakePredictor())
    response = TestClient(app_module.app).post(
        "/forecast",
        files={"file": ("candles.csv", csv_bytes, "text/csv")},
        data={"horizon": "1"},
    )
    assert response.status_code == 200
    assert "CSV upload" in response.text
    assert "32 input bars" in response.text
    assert response.text.count('class="close-cell"') == 1


@pytest.mark.parametrize("path", ["/forecast", "/forecast-market"])
def test_oversized_post_body_is_rejected_before_multipart_parse(path, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "fetch_market_candles",
        lambda *args: pytest.fail("oversized request reached market data fetch"),
    )
    boundary = "kronos-test-boundary"
    file_bytes = b"x" * (app_module.MAX_UPLOAD_BYTES + 64 * 1024)
    body = (
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="file"; filename="oversized.csv"\r\n'
        + b"Content-Type: text/csv\r\n\r\n"
        + file_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )
    response = TestClient(app_module.app).post(
        path,
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413


def test_configured_password_protects_ui_but_not_health(monkeypatch):
    auth_fixture = "kronos-test-password-with-enough-entropy"
    monkeypatch.setattr(app_module, "APP_PASSWORD", auth_fixture, raising=False)
    client = TestClient(app_module.app)
    unauthorized = client.get("/")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"].startswith("Basic")
    assert client.get("/", auth=("kronos", auth_fixture)).status_code == 200
    assert client.get("/", auth=("wrong-user", auth_fixture)).status_code == 401
    assert client.get("/health").status_code == 200


def test_configured_utf8_basic_auth_credentials_authenticate(monkeypatch):
    username = "krönos"
    auth_fixture = "pässword-with-more-than-twenty-four-characters"
    monkeypatch.setattr(app_module, "APP_USER", username, raising=False)
    monkeypatch.setattr(app_module, "APP_PASSWORD", auth_fixture, raising=False)
    response = TestClient(app_module.app).get("/", auth=(username, auth_fixture))
    assert response.status_code == 200


def test_process_rejects_wildcard_bind_without_password():
    env = os.environ.copy()
    env["KRONOS_TAILSCALE_IP"] = "0.0.0.0"
    env["KRONOS_APP_PASSWORD"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=app_module.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "KRONOS_TAILSCALE_IP" in result.stderr


def test_process_requires_password_for_tailnet_bind():
    env = os.environ.copy()
    env["KRONOS_TAILSCALE_IP"] = "100.64.0.2"
    env["KRONOS_APP_PASSWORD"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=app_module.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "KRONOS_APP_PASSWORD" in result.stderr


def test_process_allows_tailnet_bind_with_strong_password():
    env = os.environ.copy()
    env["KRONOS_TAILSCALE_IP"] = "100.64.0.2"
    env["KRONOS_APP_PASSWORD"] = "a-strong-test-password-long-enough"
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=app_module.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_request_limit_rejects_chunked_oversize_before_app():
    app_called = False
    responses = []
    messages = iter([
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"56", "more_body": True},
    ])

    async def inner(scope, receive, send):
        nonlocal app_called
        app_called = True

    async def receive():
        return next(messages)

    async def send(message):
        responses.append(message)

    scope = {"type": "http", "method": "POST", "path": "/forecast", "headers": []}
    middleware = app_module.RequestBodyLimitMiddleware(inner, max_bytes=5)
    asyncio.run(middleware(scope, receive, send))
    assert not app_called
    assert any(message.get("status") == 413 for message in responses)
