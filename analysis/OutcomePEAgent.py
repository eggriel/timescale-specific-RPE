"""
OutcomePEAgent.py
=================
Outcome-specific / Distributional RL agent using expectile regression.

Each channel i learns a different expectile τ_i of the return distribution Z(s).
Based on: Dabney et al. (2020) "A distributional code for value in dopamine-based
reinforcement learning." Nature 577, 671–675.

Model structure
---------------
* N independent channels, each with its own weight vector w_i (len = num_features)
  and expectile level τ_i ∈ (0, 1).
* Expectiles τ_i are evenly tiled across (0, 1):
      τ_i = (2i − 1) / (2N)   for i = 1, …, N
* Channel i computes value linearly:
      V_i(s) = w_i · φ(s)
* TD error for channel i:
      δ_i = r + γ V_i(s') − V_i(s)
* Asymmetric learning rates implement expectile regression:
      α_i⁺ = α · τ_i          (for δ_i ≥ 0, i.e. better-than-expected)
      α_i⁻ = α · (1 − τ_i)   (for δ_i < 0, i.e. worse-than-expected)
* Weight update:
      w_i ← w_i + lr_i · δ_i · φ(s)
        where lr_i = α_i⁺ if δ_i ≥ 0 else α_i⁻

Key properties
--------------
* Mean across channels approximates the standard scalar value:
      mean_i[V_i(s)] → E[Z(s)] = V*(s)
* Reversal point for channel i: the reward r* such that δ_i = 0.
  Optimistic channels (high τ_i) have high reversal points; pessimistic (low τ_i) low.
* DA signal returned: vector [δ_1, …, δ_N] – one scalar PE per channel.
* This is an *outcome-specific* model: heterogeneity arises in the outcome dimension
  (pessimism / optimism), NOT in the state-feature dimension.

Taxonomy (Lee et al. 2024)
--------------------------
OutcomePEAgent is an OUTCOME-SPECIFIC model. It predicts:
  - Mostly uniform cue-period responses (same features across all channels).
  - Heterogeneous outcome-period responses (different τ_i → different reversal points).
"""

import numpy as np


class OutcomePEAgent:
    def __init__(self, num_features, num_channels, lr, gamma):
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the feature vector φ(s).
        num_channels : int
            Number of parallel expectile channels N.
        lr : float
            Base learning rate α. Actual per-channel rates are α·τ_i and α·(1−τ_i).
        gamma : float
            Temporal discount factor γ ∈ [0, 1).
        """
        self.num_features = num_features
        self.num_channels = num_channels
        self.alpha = lr
        self.gamma = gamma

        # Expectile levels, evenly tiled in (0, 1)
        self.taus = np.array(
            [(2 * i - 1) / (2 * num_channels) for i in range(1, num_channels + 1)]
        )  # shape: (N,)

        # Asymmetric learning rates
        self.alpha_pos = lr * self.taus          # α_i⁺ = α · τ_i
        self.alpha_neg = lr * (1 - self.taus)    # α_i⁻ = α · (1 − τ_i)

        # Per-channel weight matrices; row i = w_i
        self.weights = np.zeros((num_channels, num_features))  # shape: (N, F)

    # ------------------------------------------------------------------
    # Value / PE computation
    # ------------------------------------------------------------------

    def val(self, state_vec):
        """
        Returns the value estimates for all N channels.

        Parameters
        ----------
        state_vec : np.ndarray, shape (num_features,)

        Returns
        -------
        np.ndarray, shape (N,)  –  V_i(s) = w_i · φ(s) for each channel i
        """
        return self.weights @ state_vec  # (N, F) @ (F,) = (N,)

    def compute_delta(self, state_vec, succ_vec, reward):
        """
        Computes the per-channel TD errors  δ_i = r + γ V_i(s') − V_i(s).

        Parameters
        ----------
        state_vec : np.ndarray, shape (num_features,)  – φ(s_t)
        succ_vec  : np.ndarray, shape (num_features,)  – φ(s_{t+1})
        reward    : float                              – r_t

        Returns
        -------
        np.ndarray, shape (N,)
        """
        v_curr = self.val(state_vec)   # (N,)
        v_next = self.val(succ_vec)    # (N,)
        return reward + self.gamma * v_next - v_curr

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, state_vec, succ_vec, reward, ret_da=True):
        """
        Update all channel weights and (optionally) return DA signals.

        The update uses asymmetric learning rates:
            w_i ← w_i + α_i^{sgn(δ_i)} · δ_i · φ(s_t)

        Parameters
        ----------
        state_vec : np.ndarray, shape (num_features,)
        succ_vec  : np.ndarray, shape (num_features,)
        reward    : float
        ret_da    : bool
            If True, return the per-channel PEs (the DA signal vector).

        Returns
        -------
        np.ndarray, shape (N,) or None
        """
        deltas = self.compute_delta(state_vec, succ_vec, reward)  # (N,)

        # Vectorised asymmetric update
        lrs = np.where(deltas >= 0, self.alpha_pos, self.alpha_neg)  # (N,)
        # w_i += lr_i * δ_i * φ(s)  →  outer product scaled by lrs * deltas
        self.weights += (lrs * deltas)[:, np.newaxis] * state_vec[np.newaxis, :]

        if ret_da:
            return deltas  # shape (N,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mean_val(self, state_vec):
        """Mean value across channels ≈ E[Z(s)] = V*(s)."""
        return np.mean(self.val(state_vec))

    def reversal_points(self):
        """
        Approximate reversal points: the converged values of each channel
        represent τ_i quantiles of the return distribution.  This helper
        returns the τ_i values (the 'reversal-point axis').
        """
        return self.taus.copy()


# ---------------------------------------------------------------------------
# Stand-alone test  (linear chain, reward at terminal state)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    num_states = 6
    N = 7
    lr = 0.05
    gamma = 0.95

    agent = OutcomePEAgent(num_states, N, lr, gamma)

    def phi(s):
        v = np.zeros(num_states)
        v[s] = 1.0
        return v

    # Stochastic reward at terminal state: r ~ Uniform[0, 2]
    for _ in range(20000):
        for s in range(num_states - 1):
            r = float(np.random.uniform(0, 2)) if s == num_states - 2 else 0.0
            agent.learn(phi(s), phi(s + 1), r)

    print("Taus:  ", np.round(agent.taus, 3))
    print("V_i(0):", np.round(agent.val(phi(0)), 4))
    print("Mean V(0) ≈ E[G_0] (should be ≈ {:.3f}):".format(
        np.mean(np.random.uniform(0, 2, 100000)) * gamma ** (num_states - 2)
    ), round(agent.mean_val(phi(0)), 4))
