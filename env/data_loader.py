# env/data_loader.py

import numpy as np
import pandas as pd
from typing import List


def generate_synthetic_btc_minute_data(
    num_days: int = 5,
    start_price: float = 50_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic minute-level BTC price data for num_days.
    This is just for early testing before you plug real BTC data.
    """
    rng = np.random.default_rng(seed)

    minutes_per_day = 24 * 60
    total_minutes = num_days * minutes_per_day

    # Simple random walk on log prices
    timestamps = pd.date_range(
        "2024-01-01", periods=total_minutes, freq="T", tz="UTC"
    )

    # Random minute returns ~ N(0, 0.0005^2) -> ~1-2% daily vol-ish
    minute_returns = rng.normal(loc=0.0, scale=0.0005, size=total_minutes)
    log_prices = np.log(start_price) + np.cumsum(minute_returns)
    prices = np.exp(log_prices)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices * (1 + rng.normal(0.0002, 0.0001, total_minutes)),
            "low": prices * (1 - rng.normal(0.0002, 0.0001, total_minutes)),
            "close": prices,
            "volume": rng.integers(1, 100, size=total_minutes),
        }
    )
    df = df.set_index("timestamp").sort_index()
    return df


def load_btc_minute_data(csv_path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load BTC minute-level OHLCV data from CSV (for later when you have real data).

    Expected columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def split_into_days(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split the BTC data into a list of daily DataFrames.
    Each DataFrame covers one calendar day.
    """
    day_groups = []
    for day, group in df.groupby(df.index.date):
        day_groups.append(group)
    return day_groups
