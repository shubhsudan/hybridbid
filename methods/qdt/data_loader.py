"""
Data loaders for QDT (Q-learning Decision Transformer).

Three stages need different data views:

Stage 1 (CQL critic):
  PostbreakDataset — same as DQL: (obs, act, rew, next_obs, done)
  Rewards recomputed to physical-$ via shared utility.

Stage 2 (RTG relabeling):
  Uses trained CQL critic to produce Q(s,a) for every transition.
  Saves relabeled dataset to methods/qdt/dataset_relabeled.npz.

Stage 3 (Decision Transformer):
  SequenceDataset — returns fixed-length K-step context windows
  (rtg_relabeled, obs, action) used to train the DT autoregressively.
  RTG at each position uses the Q-value from Stage 2, not MC returns.

Action convention: p.u. throughout training; de-normalized to physical MW
at eval time by policy wrapper.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys, os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from methods._shared.reward_recompute import recompute_rewards

OBS_DIM = 32 * 12 + 14   # 398
ACT_DIM = 6
K = 20  # DT context length (steps, = 100 min)


def flatten_obs(price_history: np.ndarray, static_features: np.ndarray) -> np.ndarray:
    return np.concatenate([price_history.reshape(-1), static_features], axis=-1)


# ── Stage 1: CQL data ───────────────────────────────────────────────���──────

class PostbreakDataset(Dataset):
    """
    Flat (obs, act, rew, next_obs, done, next_act, sarsa_done) transitions for CQL.

    Physical-$ rewards recomputed at load.

    next_act / sarsa_done: in-sample SARSA bootstrap.
      next_act[i]   = actions[i+1] for non-boundary transitions.
      sarsa_done[i] = 1.0 at CT-midnight truncated boundaries and at the final transition;
                      zeros the γ·Q(s', a') bootstrap term so it doesn't cross episode
                      boundaries. This is the ONLY place truncated zeros the bootstrap —
                      it applies only to the next-action lookup, not to the done flag for
                      the current transition.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        rewards = recompute_rewards(dict(data))

        ph  = data["price_history"]
        sf  = data["static_features"]
        nph = data["next_price_history"]
        nsf = data["next_static_features"]
        acts = data["actions"].astype(np.float32)

        obs      = np.concatenate([ph.reshape(len(ph), -1), sf],   axis=1).astype(np.float32)
        next_obs = np.concatenate([nph.reshape(len(nph), -1), nsf], axis=1).astype(np.float32)

        N = len(acts)

        # next_act[i] = acts[i+1]; last row gets zeros (handled by sarsa_done)
        next_act = np.empty_like(acts)
        next_act[:-1] = acts[1:]
        next_act[-1]  = np.zeros(ACT_DIM, dtype=np.float32)

        # sarsa_done[i] = 1 at truncated boundaries and at the final step
        trunc = data["truncateds"]  # bool (N,); True at last step of each CT-day episode
        sarsa_done = trunc.astype(np.float32)
        sarsa_done[-1] = 1.0  # last transition in dataset has no valid next_act

        self.obs        = torch.from_numpy(obs)
        self.act        = torch.from_numpy(acts)
        self.rew        = torch.from_numpy(rewards)
        self.next_obs   = torch.from_numpy(next_obs)
        self.done       = torch.zeros(N, dtype=torch.float32)  # no terminal states
        self.next_act   = torch.from_numpy(next_act)
        self.sarsa_done = torch.from_numpy(sarsa_done)

        self.n = N
        print(f"[PostbreakDataset/QDT] {self.n:,} transitions  "
              f"rew mean={rewards.mean():.3f}  std={rewards.std():.3f}  "
              f"sarsa_boundaries={int(sarsa_done.sum())}")

    def __len__(self):  return self.n
    def __getitem__(self, i):
        return (self.obs[i], self.act[i], self.rew[i], self.next_obs[i],
                self.done[i], self.next_act[i], self.sarsa_done[i])


def make_cql_loader(npz_path: str, batch_size: int = 256) -> DataLoader:
    return DataLoader(PostbreakDataset(npz_path), batch_size=batch_size,
                      shuffle=True, num_workers=0, pin_memory=True, drop_last=True)


# ── Stage 2: RTG relabeling ────────────────────────────────────────────────

def relabel_rtg(npz_path: str, critic, device: torch.device,
                batch_size: int = 512) -> np.ndarray:
    """
    Compute relabeled RTG for every transition: rtg[t] = Q(s_t, a_t).

    Uses the CQL critic's q_min (conservative lower bound) as the target RTG.
    Returns (N,) float32 array of per-step Q-value labels.
    """
    data = np.load(npz_path, allow_pickle=False)
    ph   = data["price_history"]
    sf   = data["static_features"]
    acts = data["actions"]
    N    = len(acts)

    obs_flat = np.concatenate([ph.reshape(N, -1), sf], axis=1).astype(np.float32)
    qtgt = np.empty(N, dtype=np.float32)

    critic.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end  = min(start + batch_size, N)
            o    = torch.from_numpy(obs_flat[start:end]).to(device)
            a    = torch.from_numpy(acts[start:end]).to(device)
            q1, q2 = critic(o, a)
            qtgt[start:end] = torch.min(q1, q2).squeeze(1).cpu().numpy()

    print(f"[RTG relabel] Q-values: mean={qtgt.mean():.3f}  std={qtgt.std():.3f}  "
          f"min={qtgt.min():.3f}  max={qtgt.max():.3f}")
    return qtgt


def save_relabeled(npz_path: str, rtg_values: np.ndarray, out_path: str) -> None:
    """Save original trajectory + relabeled RTG to a new NPZ."""
    data = dict(np.load(npz_path, allow_pickle=False))
    data["rtg"] = rtg_values.astype(np.float32)
    np.savez(out_path, **data)
    print(f"[RTG relabel] Saved relabeled dataset to {out_path}")


# ── Stage 3: Decision Transformer sequence dataset ─────────────────────────

class SequenceDataset(Dataset):
    """
    K-step context windows for DT training.

    Each sample is a window of K consecutive steps (within the same episode/day):
      rtg    : (K,)   relabeled Q-value at each step
      obs    : (K, 398)
      act    : (K, 6)
      target : (K, 6) action labels (same as act, shifted by 0)

    Windows do NOT cross CT-midnight episode boundaries (truncateds=True marks
    end-of-day; a new window cannot include steps from the next day).

    During eval, the DT slides a K=20 window over the continuous 54-day run.
    """

    def __init__(self, relabeled_path: str):
        data = np.load(relabeled_path, allow_pickle=False)

        ph   = data["price_history"]
        sf   = data["static_features"]
        acts = data["actions"].astype(np.float32)
        rtg  = data["rtg"].astype(np.float32)
        trunc = data["truncateds"]

        N   = len(acts)
        obs = np.concatenate([ph.reshape(N, -1), sf], axis=1).astype(np.float32)

        # Build list of valid window start indices:
        # A window [i, i+K) is valid if no truncated boundary falls strictly inside it.
        # Truncated at position t means end-of-day; window must not straddle t+1.
        trunc_indices = set(np.where(trunc)[0].tolist())

        self.windows: list[int] = []
        for i in range(N - K + 1):
            # Window spans [i, i+K). Truncated at position t means step t is last of episode.
            # A straddle occurs if any t in trunc_indices is in [i, i+K-1).
            # (i+K-1 is the last step of the window, which can be truncated)
            straddled = any(t in trunc_indices for t in range(i, i + K - 1))
            if not straddled:
                self.windows.append(i)

        self.obs  = obs
        self.acts = acts
        self.rtg  = rtg

        print(f"[SequenceDataset] {N:,} transitions → {len(self.windows):,} valid K={K} windows")
        # RTG stats (these are Q-values in physical $)
        window_rtg0 = np.array([rtg[w] for w in self.windows])
        print(f"  RTG(t=0) per window: mean={window_rtg0.mean():.2f}  "
              f"P50={np.median(window_rtg0):.2f}  P95={np.percentile(window_rtg0, 95):.2f}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        i = self.windows[idx]
        return (
            torch.from_numpy(self.rtg[i:i+K]),        # (K,) target Q-values
            torch.from_numpy(self.obs[i:i+K]),         # (K, 398)
            torch.from_numpy(self.acts[i:i+K]),        # (K, 6)
        )


def make_sequence_loader(relabeled_path: str, batch_size: int = 64) -> DataLoader:
    return DataLoader(SequenceDataset(relabeled_path), batch_size=batch_size,
                      shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
