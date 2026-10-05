from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import get_predictor


def main() -> None:
    timestamps = pd.date_range("2025-01-02 09:30", periods=64, freq="min")
    steps = np.arange(len(timestamps), dtype=np.float32)
    close = 100.0 + 0.015 * steps + 0.2 * np.sin(steps / 4.0)
    candles = pd.DataFrame({
        "open": close - 0.02,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": 1000.0 + 5.0 * steps,
    })
    future = pd.Series(pd.date_range(timestamps[-1] + pd.Timedelta(minutes=1), periods=2, freq="min"))
    result = get_predictor().predict(
        df=candles,
        x_timestamp=pd.Series(timestamps),
        y_timestamp=future,
        pred_len=2,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    values = result[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
    assert result.shape[0] == 2
    assert np.isfinite(values).all()
    print(f"SMOKE_OK rows={result.shape[0]} all_finite=true")


if __name__ == "__main__":
    main()
