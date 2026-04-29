"""Unit tests for methods/_shared/reward_recompute.py."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.methods._shared.reward_recompute import recompute_rewards, P_MAX, DT


def _make_trajectory(actions, price_row):
    """Build minimal trajectory dict: all steps share the same current-step price."""
    N = len(actions)
    price_history = np.zeros((N, 32, 12), dtype=np.float32)
    price_history[:, -1, :] = price_row  # last row = current step
    return {"actions": np.array(actions, dtype=np.float32), "price_history": price_history}


class TestRecomputeRewards:

    def test_idle_step_pure_as(self):
        """Idle step (zero energy) with AS bids: reward = AS physical $."""
        # 1 MW regup at $50/MWh, no energy action
        # reward = 0 (energy) + 1.0 × 50 × DT = 50 × 5/60 ≈ 4.1667
        price_row = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action = np.array([0.0, 1.0 / P_MAX, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 1 MW regup normalized
        traj = _make_trajectory([action], price_row)
        rewards = recompute_rewards(traj)
        expected = 1.0 * 50.0 * DT
        assert abs(float(rewards[0]) - expected) < 1e-4, f"got {rewards[0]:.6f}, want {expected:.6f}"

    def test_pure_discharge(self):
        """Full discharge at 10 MW, no AS: reward = +10 × rt_lmp × DT."""
        rt_lmp = 100.0
        price_row = np.zeros(12, dtype=np.float32)
        price_row[0] = rt_lmp
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # p_energy = 1.0 p.u.
        traj = _make_trajectory([action], price_row)
        rewards = recompute_rewards(traj)
        expected = P_MAX * rt_lmp * DT   # 10 × 100 × 5/60 = 83.333
        assert abs(float(rewards[0]) - expected) < 1e-3, f"got {rewards[0]:.4f}, want {expected:.4f}"

    def test_pure_charge_negative_energy_rev(self):
        """Charging (negative p_energy): energy rev is negative (we pay to charge)."""
        rt_lmp = 30.0
        price_row = np.zeros(12, dtype=np.float32)
        price_row[0] = rt_lmp
        action = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 5 MW charge
        traj = _make_trajectory([action], price_row)
        rewards = recompute_rewards(traj)
        expected = -0.5 * P_MAX * rt_lmp * DT  # -5 × 30 × 5/60 = -12.5
        assert abs(float(rewards[0]) - expected) < 1e-3, f"got {rewards[0]:.4f}, want {expected:.4f}"

    def test_charge_with_as_mixed_sign(self):
        """Charge + AS: energy rev negative, AS rev positive. Total may be either sign."""
        rt_lmp = 5.0         # cheap energy
        regup_price = 25.0   # good AS price
        price_row = np.zeros(12, dtype=np.float32)
        price_row[0] = rt_lmp
        price_row[1] = regup_price
        # charge 5 MW + 5 MW regup
        action = np.array([-0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        traj = _make_trajectory([action], price_row)
        rewards = recompute_rewards(traj)
        energy_part = -0.5 * P_MAX * rt_lmp * DT        # -5 × 5 × 5/60 = -2.0833
        as_part = 0.5 * P_MAX * regup_price * DT        # 5 × 25 × 5/60 = 10.4167
        expected = energy_part + as_part                 # +8.3333
        assert float(rewards[0]) > 0, "total should be positive (AS > energy cost)"
        assert abs(float(rewards[0]) - expected) < 1e-3

    def test_sign_flip_case(self):
        """
        Physical revenue is negative but stored reward was positive (the sign-flip bug).
        Recomputed reward must be negative (correct).

        Replicates the pattern from REWARD_CONVENTION.md idx=16811:
        charging at high price → physical rev negative.
        """
        rt_lmp = 400.0  # expensive; charging here is costly
        price_row = np.zeros(12, dtype=np.float32)
        price_row[0] = rt_lmp
        action = np.array([-0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 8 MW charge
        traj = _make_trajectory([action], price_row)
        rewards = recompute_rewards(traj)
        expected = -0.8 * P_MAX * rt_lmp * DT  # -8 × 400 × 5/60 = -266.67
        assert float(rewards[0]) < 0, "physical revenue must be negative when charging at $400"
        assert abs(float(rewards[0]) - expected) < 0.1

    def test_batch_computation_matches_per_sample(self):
        """Batch result matches calling per-sample."""
        rng = np.random.default_rng(99)
        N = 50
        actions = rng.uniform(-1, 1, (N, 6)).astype(np.float32)
        actions[:, 1:] = np.abs(actions[:, 1:])  # AS non-negative
        price_history = np.zeros((N, 32, 12), dtype=np.float32)
        price_history[:, -1, 0] = rng.uniform(10, 500, N)     # rt_lmp
        price_history[:, -1, 1:6] = rng.uniform(0, 50, (N, 5))  # rt_mcpc

        traj = {"actions": actions, "price_history": price_history}
        batch_rewards = recompute_rewards(traj)

        for i in range(N):
            single = _make_trajectory([actions[i]], price_history[i:i+1, -1, :])
            single_rew = recompute_rewards(single)
            assert abs(float(batch_rewards[i]) - float(single_rew[0])) < 1e-5, \
                f"batch vs single mismatch at i={i}"

    def test_against_real_trajectory_idle_steps(self):
        """
        On real trajectory data: truly zero-energy steps must have stored ≈ recomputed.
        At these steps, stored reward = pure AS physical $ = recomputed reward.

        Threshold 1e-4 p.u. = 0.001 MW — avoids MILP numerical-noise residuals
        (residuals at 1e-3 p.u. contribute ~$0.1 at high prices, causing false failures).
        """
        data_path = "data/expert_trajectories/receding_horizon_postbreak_train.npz"
        try:
            data = np.load(data_path, allow_pickle=False)
        except FileNotFoundError:
            pytest.skip(f"trajectory file not found: {data_path}")

        recomputed = recompute_rewards(dict(data))
        stored = data["rewards"]
        actions = data["actions"]

        idle_mask = np.abs(actions[:, 0]) < 1e-4
        assert idle_mask.sum() > 0, "no idle steps found"

        diff = np.abs(recomputed[idle_mask] - stored[idle_mask])
        assert diff.max() < 1e-3, \
            f"idle steps: max|recomputed - stored| = {diff.max():.6f} (should be ~0)"

    def test_against_real_trajectory_physical_total(self):
        """
        Total recomputed rewards should match expected physical total ($116,511 ≈ $116,669).
        Allows 5% tolerance for timing/rounding differences.
        """
        data_path = "data/expert_trajectories/receding_horizon_postbreak_train.npz"
        try:
            data = np.load(data_path, allow_pickle=False)
        except FileNotFoundError:
            pytest.skip(f"trajectory file not found: {data_path}")

        recomputed = recompute_rewards(dict(data))
        total = float(recomputed.sum())
        expected = 116_669.0  # MILP-internal $116,669 from CLAUDE.md
        tolerance = 0.05 * expected
        assert abs(total - expected) < tolerance, \
            f"recomputed total ${total:.0f} deviates > 5% from expected ${expected:.0f}"

    def test_output_dtype_and_shape(self):
        """Output must be float32 with shape (N,)."""
        actions = np.zeros((10, 6), dtype=np.float32)
        ph = np.zeros((10, 32, 12), dtype=np.float32)
        rewards = recompute_rewards({"actions": actions, "price_history": ph})
        assert rewards.dtype == np.float32, f"expected float32, got {rewards.dtype}"
        assert rewards.shape == (10,), f"expected (10,), got {rewards.shape}"

    def test_zero_prices_zero_reward(self):
        """Zero prices → zero reward regardless of action."""
        actions = np.random.rand(5, 6).astype(np.float32)
        ph = np.zeros((5, 32, 12), dtype=np.float32)
        rewards = recompute_rewards({"actions": actions, "price_history": ph})
        assert np.allclose(rewards, 0.0), "zero prices should yield zero reward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
