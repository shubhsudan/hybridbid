"""
Transformer price forecaster for MILP+forecaster baseline (Method 1).

Input:  price_history (B, 32, 12) — last 32 × 5-min price observations
Output: (B, 288, 6) — next 24h forecast of [rt_lmp, rt_mcpc × 5]

Architecture:
  - Linear input embedding: 12 → 128 + sinusoidal positional encoding
  - 4-layer TransformerEncoder (8 heads, d_model=128, ff=512, dropout=0.1)
  - Global average pool → (B, 128)
  - MLP head: 128 → 512 → 1728, reshape → (B, 288, 6)

Price normalization (sign-preserving log1p):
  - Energy prices (col 0 = rt_lmp): sign(x) * log1p(|x|)
  - AS prices (cols 1-5 = rt_mcpc_*): log1p(max(0, x))
  - All 12 input features use same scheme (cols 0,6 energy; rest AS)
  - Inverse available for converting forecasts back to physical $/MWh
"""

import math

import numpy as np
import torch
import torch.nn as nn

HIST_LEN    = 32
FUTURE_LEN  = 288
N_IN_FEAT   = 12
N_OUT_FEAT  = 6    # rt_lmp + 5 rt_mcpc
D_MODEL     = 128
N_HEADS     = 8
N_LAYERS    = 4
FF_DIM      = 512
DROPOUT     = 0.1

# Column indices in price_history that are energy prices (can be negative)
_ENERGY_COLS_12 = (0, 6)   # rt_lmp, dam_spp


# ── Price transforms ──────────────────────────────────────────────────────────

def price_transform_12(x: np.ndarray) -> np.ndarray:
    """(*, 12) float32 → log1p-transformed, same shape."""
    out = np.empty_like(x, dtype=np.float32)
    for j in range(12):
        if j in _ENERGY_COLS_12:
            out[..., j] = np.sign(x[..., j]) * np.log1p(np.abs(x[..., j]))
        else:
            out[..., j] = np.log1p(np.clip(x[..., j], 0.0, None))
    return out


def price_transform_6(x: np.ndarray) -> np.ndarray:
    """(*, 6) float32 [rt_lmp, 5 rt_mcpc] → log1p-transformed."""
    out = np.empty_like(x, dtype=np.float32)
    out[..., 0] = np.sign(x[..., 0]) * np.log1p(np.abs(x[..., 0]))     # rt_lmp
    out[..., 1:] = np.log1p(np.clip(x[..., 1:], 0.0, None))              # rt_mcpc
    return out


def price_inverse_transform_6(x: np.ndarray) -> np.ndarray:
    """Inverse of price_transform_6: log-space → physical $/MWh."""
    out = np.empty_like(x, dtype=np.float32)
    out[..., 0] = np.sign(x[..., 0]) * np.expm1(np.abs(x[..., 0]))     # rt_lmp
    out[..., 1:] = np.expm1(np.clip(x[..., 1:], -30.0, 30.0))           # rt_mcpc
    out[..., 1:] = np.clip(out[..., 1:], 0.0, None)
    return out


# ── Positional encoding ───────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── Transformer forecaster ────────────────────────────────────────────────────

class PriceTransformer(nn.Module):
    """
    Transformer-based RT price forecaster.
    Input:  (B, 32, 12) log-transformed price history
    Output: (B, 288, 6) log-transformed forecast [rt_lmp, 5 rt_mcpc]
    """

    def __init__(
        self,
        hist_len:   int = HIST_LEN,
        n_in_feat:  int = N_IN_FEAT,
        future_len: int = FUTURE_LEN,
        n_out_feat: int = N_OUT_FEAT,
        d_model:    int = D_MODEL,
        n_heads:    int = N_HEADS,
        n_layers:   int = N_LAYERS,
        ff_dim:     int = FF_DIM,
        dropout:    float = DROPOUT,
    ):
        super().__init__()
        self.future_len  = future_len
        self.n_out_feat  = n_out_feat

        self.input_proj = nn.Linear(n_in_feat, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=hist_len + 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        out_dim = future_len * n_out_feat
        self.head = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 32, 12) float32 — log-transformed price history
        returns: (B, 288, 6) float32 — log-transformed price forecast
        """
        h = self.pos_enc(self.input_proj(x))    # (B, 32, 128)
        h = self.encoder(h)                      # (B, 32, 128)
        h = h.mean(dim=1)                        # (B, 128) global avg pool
        out = self.head(h)                       # (B, future_len * n_out_feat)
        return out.view(-1, self.future_len, self.n_out_feat)


def predict_prices(
    model: PriceTransformer,
    price_history_raw: np.ndarray,   # (32, 12) physical $/MWh
    device: str = "cpu",
) -> np.ndarray:
    """
    Run inference: raw (32, 12) → physical (288, 6) forecast in $/MWh.
    Returns the 6 RT price columns: [rt_lmp, rt_mcpc_regup, ..., rt_mcpc_nsrs].
    """
    model.eval()
    x_t  = price_transform_12(price_history_raw)            # (32, 12) log-space
    x_t  = torch.from_numpy(x_t).unsqueeze(0).to(device)   # (1, 32, 12)
    with torch.no_grad():
        pred_log = model(x_t).squeeze(0).cpu().numpy()      # (288, 6) log-space
    return price_inverse_transform_6(pred_log)               # (288, 6) physical $/MWh
