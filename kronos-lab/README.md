# Kronos Lab

A small Tailscale-only web workbench for testing the upstream Kronos-small candle forecaster. This is a separate application, **not a fork** of Kronos: the original repository is pinned as a Git submodule under `vendor/Kronos`.

## Current scope

- Fetch U.S.-listed ticker candles by symbol, interval, and requested count, or upload chronological OHLCV CSV data.
- Pinned upstream source, model, and tokenizer revisions for reproducibility.
- CPU inference; pretrained weights download on the first forecast and persist in the cache volume.
- One greedy forecast path. No confidence interval, decision model, broker, or order execution is connected.
- Market-fetch forecasts use future timestamps on the XNYS session calendar, including holidays and early closes; CSV forecasts extrapolate from median input spacing and do not infer exchange sessions.

## CSV format

Required columns: `timestamp` (or `datetime` / `date`), `open`, `high`, `low`, `close`. Optional: `volume`, `amount`. At least 32 rows are required; the last 512 bars are sent to the model. The upload is processed in memory and is not saved by the app.

## Live candle fetch

The dashboard defaults to `MCD`, `1h`, and 240 bars. It supports `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, and `1wk`, with a requested count of 32–512. The forecast horizon is independently selectable from 1–128 bars.

Live fetching uses Yahoo Finance’s public chart endpoint without an API key. It is a best-effort prototype feed, not a service-guaranteed data source. The app accepts only Yahoo-confirmed U.S.-listed USD equities and uses XNYS session schedules to filter non-session daily/intraday candles, omit incomplete bars, and account for scheduled early closes while retaining valid candles from those sessions. 4h candles are built from all expected hourly constituents on the session grid; the closing-session bucket can be shorter. If the provider cannot return the requested number of complete bars, the app reports an error rather than silently using fewer. Market forecasts use calendar-aware future timestamps; CSV upload remains available and uses median-spacing timestamp extrapolation.

## Run locally with Docker

From the root of a clone of the Kronos repository, initialize the pinned model-source submodule once, then enter this app directory:

```sh
git submodule update --init --recursive
cd kronos-lab
docker compose up --build
```

By default, Compose binds to loopback only (`127.0.0.1:8128`) and no app password is needed. Use that direct command for local-only access.

For tailnet access, create an untracked `.env` file in `kronos-lab/` containing a unique `KRONOS_APP_PASSWORD` of at least 24 characters (and optionally `KRONOS_APP_USER`). Ensure the host is connected to Tailscale, then launch from this directory with:

```sh
./scripts/run-tailnet.sh up --build -d
```

The launcher reads the host's actual address with `tailscale ip -4`, rejects non-Tailscale IPv4 values and any conflicting `KRONOS_TAILSCALE_IP` environment setting, then passes the verified address to Compose. It requires the host Tailscale CLI, Docker Compose, and Python 3. The app also rejects wildcard/public binds and refuses a non-loopback bind without the strong password; UI and forecast routes use HTTP Basic Auth (the health endpoint is exempt for container checks). The app's range check is defense-in-depth, not proof that an address belongs to Tailscale: use the launcher for tailnet exposure, not a manually overridden Compose bind. From a tailnet device, open `http://<host-tailscale-ip>:8128`.

## Checks

```sh
docker compose run --rm --no-deps --entrypoint python kronos-lab -m pytest -q
docker compose run --rm --no-deps kronos-lab python /app/tests/smoke_model.py
```

The first command runs parser and route unit tests; the second downloads (if needed) and exercises the pinned model using synthetic candles.

## Upstream and weights

- Upstream code submodule: `shiyu-coder/Kronos` at the gitlink recorded in this repository.
- Model: `NeoQuasar/Kronos-small`, revision `901c26c1332695a2a8f243eb2f37243a37bea320`.
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`, revision `0e0117387f39004a9016484a186a908917e22426`.
- **Model cache:** the named Compose volume `huggingface-cache`; weights are not committed here.

## Safety / limitations

This is an experiment UI, not financial advice or a trading system. A forecast is not a calibrated probability of profit. Validate on chronological holdouts with transaction costs and realistic execution before using any result. JEV, Laya, and any local classifier remain intentionally disconnected until their decision task, data boundary, labels, and evaluation are defined.
