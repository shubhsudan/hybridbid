"""
Dataset for forecaster training.

Each sample: (price_history (32, 12), future_rt_prices (288, 6))
Both in log-transformed space (see forecaster.price_transform_*).

Training split: all data strictly before 2026-01-01 CT.
Val split:      2025-12-01 to 2025-12-31 CT (1 month held out from training).
T-60 window:    2026-01-01 to 2026-02-23 — never used for train or val.
"""

from __future__ import annotations

import glob
import os
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .forecaster import price_transform_12, price_transform_6

# Column ordering matches experiments/prepare_postbreak.py exactly
PRICE_COLS = [
    "rt_lmp",
    "rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs",
    "dam_spp",
    "dam_as_regup", "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs",
]
# The 6 output features the forecaster predicts (first 6 of PRICE_COLS)
OUTPUT_COLS = PRICE_COLS[:6]

HIST_LEN   = 32
FUTURE_LEN = 288

# Cut dates (CT midnight ≡ UTC 06:00 of same date)
TRAIN_END_CT = date(2025, 11, 30)   # inclusive train end
VAL_END_CT   = date(2025, 12, 31)   # inclusive val end (Dec 2025)
T60_START_CT = date(2026, 1, 1)


def _load_price_series(data_dir: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load and merge energy + AS price parquets, return (array, index).

    Returns
    -------
    arr   : (N, 12) float32, log1p-NOT-yet-applied (raw $/MWh)
    index : UTC DatetimeIndex of length N
    """
    def _read(subdir):
        files = sorted(glob.glob(os.path.join(data_dir, subdir, "*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in files]).sort_index()

    ep = _read("energy_prices")
    ap = _read("as_prices")

    if "is_post_rtcb" in ep.columns:
        ep = ep.drop(columns=["is_post_rtcb"])
    if "is_post_rtcb" in ap.columns:
        ap = ap.drop(columns=["is_post_rtcb"])

    merged = ep.join(ap, how="outer")
    merged = merged[PRICE_COLS]
    merged[PRICE_COLS] = merged[PRICE_COLS].fillna(0.0)

    return merged.values.astype(np.float32), merged.index


def _ct_midnight_utc_cutoff(ct_date: date) -> pd.Timestamp:
    """CT midnight for ct_date = UTC 06:00 of that date."""
    return pd.Timestamp(ct_date.year, ct_date.month, ct_date.day, 6, 0, 0, tz="UTC")


class PriceForecastDataset(Dataset):
    """
    Sliding-window forecast dataset.

    Each sample index i → observation i, which yields:
      obs_raw  : raw (32, 12) price window ending at valid_indices[i]
      tgt_raw  : raw (288, 6) future RT prices starting at valid_indices[i]+1
    Both converted to log-space at __getitem__ time.
    """

    def __init__(
        self,
        price_arr:     np.ndarray,          # (N, 12) float32 raw prices
        valid_indices: np.ndarray,          # 1-D int array of valid start positions
    ):
        self.price_arr     = price_arr
        self.valid_indices = valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.valid_indices[i]

        # History window: [idx - HIST_LEN + 1 : idx + 1]
        h_start = idx - HIST_LEN + 1
        hist_raw = self.price_arr[h_start : idx + 1]          # (32, 12)
        hist_log = price_transform_12(hist_raw)

        # Future window: [idx + 1 : idx + 1 + FUTURE_LEN], first 6 cols only
        tgt_raw  = self.price_arr[idx + 1 : idx + 1 + FUTURE_LEN, :6]   # (288, 6)
        tgt_log  = price_transform_6(tgt_raw)

        return (
            torch.from_numpy(hist_log),   # (32, 12)
            torch.from_numpy(tgt_log),    # (288, 6)
        )


def make_datasets(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    Train:  indices where obs falls in [2020-01-01, 2025-11-30] CT
    Val:    indices where obs falls in [2025-12-01, 2025-12-31] CT

    T-60 window (2026-01-01 → 2026-02-23) is excluded from both.
    """
    print("[forecast_dataset] Loading full price series...")
    price_arr, utc_index = _load_price_series(data_dir)
    N = len(price_arr)
    print(f"[forecast_dataset] {N} rows, {utc_index[0]} to {utc_index[-1]}")

    # Min index that can form a full history window
    min_idx = HIST_LEN - 1

    # Max index that can form a full future window
    max_idx = N - FUTURE_LEN - 1

    # CT cutoff timestamps in UTC
    train_end_utc = _ct_midnight_utc_cutoff(date(2025, 12, 1))  # train: obs < Dec 1 CT
    val_end_utc   = _ct_midnight_utc_cutoff(T60_START_CT)        # val: obs < Jan 1, 2026 CT

    # Obs timestamp = utc_index[idx] (the last step of the history window)
    obs_ts = utc_index

    train_mask = (
        (np.arange(N) >= min_idx)
        & (np.arange(N) <= max_idx)
        & (obs_ts < train_end_utc)
    )
    val_mask = (
        (np.arange(N) >= min_idx)
        & (np.arange(N) <= max_idx)
        & (obs_ts >= train_end_utc)
        & (obs_ts < val_end_utc)
    )

    train_indices = np.where(train_mask)[0]
    val_indices   = np.where(val_mask)[0]

    print(f"[forecast_dataset] Train samples: {len(train_indices):,}  Val samples: {len(val_indices):,}")

    rng = np.random.default_rng(42)
    rng.shuffle(train_indices)

    train_ds = PriceForecastDataset(price_arr, train_indices)
    val_ds   = PriceForecastDataset(price_arr, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=False, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def load_t60_price_array(data_dir: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Return raw price array and index for the T-60 eval window + 32-step buffer.
    Used by the policy to extract obs price_history at inference time.
    This is essentially what prepare_postbreak.load_data provides.
    """
    price_arr, utc_index = _load_price_series(data_dir)
    # Filter to T-60 window with 32-step look-back buffer (Dec 28 onward)
    buffer_start = pd.Timestamp("2025-12-28 00:00:00+00:00")
    mask = utc_index >= buffer_start
    return price_arr[mask], utc_index[mask]
