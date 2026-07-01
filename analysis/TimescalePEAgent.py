"""
TimescalePEAgent.py
===================
Custom β-weighted RPE agent with N independent value channels, each learning
a linear combination of J input features.  Two-timescale learning: weights W
update slowly; channel gains β update fast.

Geometry
--------
    J = n_features   — dimension of the shared feature vector φ_t ∈ ℝ^J
    N = n_values     — number of parallel value channels

Each channel i maintains a weight vector  w_i ∈ ℝ^J  (row i of the N×J
weight matrix W).  Channel i computes:

    V_i(s) = w_i · φ(s)         (dot product over all J features)

Total β-weighted value:

    V_tot(s) = Σ_i β_i V_i(s) = β · (W φ(s))

This is a many-to-many (J→N) mapping: every feature contributes to every
value channel, each with its own learned weight.  This differs from
VectorRPEAgent (one-to-one, N=J, scalar w_i) and OutcomePEAgent (N<J, same
W for all channels, heterogeneity only in asymmetric learning rates).

Biological interpretation
--------------------------
N striatal subregions each receive the same cortical feature vector φ_t
(via topographic-but-overlapping projections), but learn a different linear
readout w_i.  The β_i gains reflect the projection strength from each
subregion onto the DA population.  DA reports δ_total = Σ_i β_i δ_i, the
β-weighted sum of per-subregion prediction errors.

Per-channel prediction error
-----------------------------
    δ_i = r_t / (β_i N)  +  w_i · (γ φ_{t+1} − φ_t)

Consistency:  Σ_i β_i δ_i = δ_total = r + γ V_tot(t+1) − V_tot(t)  ✓

Proof:
    Σ_i β_i δ_i
    = Σ_i [ r/N  +  β_i w_i·(γφ_{t+1}−φ_t) ]
    = r  +  (β·W)·(γφ_{t+1}−φ_t)
    = r  +  γ V_tot(t+1)  −  V_tot(t)   □

Two-timescale learning
----------------------
Slow — weight update (local, one outer product per step):
    W ← W + α · outer(δ_feat, φ_t)
    i.e. w_i ← w_i + α · δ_i · φ_t   for each channel i

This is a LOCAL update: channel i uses only its own δ_i.  Because δ_i
involves the full dot product  w_i·(γφ_{t+1}−φ_t)  (not just one feature),
value propagates correctly across multiple time steps even with one-hot
features.  Convergence to V* holds with standard TD conditions.

Fast — β update (gradient of ½δ_total² w.r.t. β_i):
    ∂L/∂β_i = −δ_total · V_i(s_t)
    β_i ← β_i + α_β · δ_total · V_i_old
    (V_i_old computed with weights BEFORE the W update)

β is clipped to beta_min then renormalised so Σ β_i = N.
"""

import numpy as np


class TimescalePEAgent:
    def __init__(self, n_features, n_values, lr, gamma, delta_beta=None,
                 betas=None, lr_beta=None,
                 beta_min=1e-3, normalize_betas=True, weight_noise=0.01, beta_noise=0.1):
        """
        Parameters
        ----------
        n_features : int  — J, feature-vector dimension
        n_values   : int  — N, number of parallel value channels
        lr         : float — learning rate for W  (slow timescale)
        gamma      : float — discount factor
        betas      : array-like (N,) or None — initial channel gains
                     (None → all ones)
        lr_beta    : float or None — learning rate for β  (None → 2×lr)
        beta_min   : float — hard lower bound on each β_i
        normalize_betas : bool — keep Σ β_i = N after every β update
        weight_noise : float — std of Gaussian noise added to initial W (default 0.01).
        beta_noise   : float — std of noise added to initial β (default 0.1).
            This is the primary symmetry-breaker for β: channels get different
            gains from step one, so the gradient δ_total·V_i is non-uniform
            immediately.  weight_noise alone forces β to wait until weight
            differences grow into distinct V_i — which can take hundreds of
            episodes.  Set beta_noise=0 to start from a perfectly flat prior.
        """
        self.n_features      = n_features   # J
        self.n_values        = n_values     # N
        self.alpha           = lr
        self.alpha_beta      = (10.0 * lr) if lr_beta is None else float(lr_beta)
        self.gamma           = gamma
        self.beta_min        = beta_min
        self.normalize_betas = normalize_betas

        # Weight matrix W ∈ ℝ^{N×J}; row i is w_i
        if weight_noise > 0:
            self.weights = np.random.randn(n_values, n_features) * weight_noise
        else:
            self.weights = np.zeros((n_values, n_features))

        # Channel gains β ∈ ℝ^N
        # Start from explicit values or ones, then add noise to break symmetry.
        if betas is None:
            self.betas = np.ones(n_values, dtype=float)
        else:
            self.betas = np.asarray(betas, dtype=float).copy()
            if self.betas.shape != (n_values,):
                raise ValueError(f"betas must have shape ({n_values},), got {self.betas.shape}")
            if np.any(self.betas <= 0):
                raise ValueError("All initial β_i must be strictly positive.")

        if beta_noise > 0:
            # Use abs() so all values stay positive; normalise so Σβ_i = N
            # self.betas = np.random.randn(n_values) * beta_noise

            self.betas += np.abs(np.random.randn(n_values)) * beta_noise
            np.clip(self.betas, self.beta_min, None, out=self.betas)
            self.betas *= n_values / self.betas.sum()

    # ── Value ──────────────────────────────────────────────────────────────────

    def val(self, state_vec):
        """
        Scalar total value:  V_tot = β · (W φ).

        Parameters
        ----------
        state_vec : (n_features,)

        Returns
        -------
        float
        """
        return float(np.dot(self.betas, self.weights @ state_vec))

    def val_per_channel(self, state_vec):
        """
        Per-channel values  V_i = w_i · φ,  shape (N,).

        Parameters
        ----------
        state_vec : (n_features,)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        return self.weights @ state_vec   # (N, J) @ (J,) = (N,)

    # ── Prediction errors ──────────────────────────────────────────────────────

    def compute_delta_feat(self, state_vec, succ_vec, reward):
        """
        Per-channel prediction errors  δ_i = r/(β_i N) + w_i·(γφ'−φ),  shape (N,).

        Parameters
        ----------
        state_vec : (n_features,)
        succ_vec  : (n_features,)
        reward    : float

        Returns
        -------
        np.ndarray, shape (N,)
        """
        N             = self.n_values
        reward_term   = reward / (self.betas * N)                         # (N,)
        reward_term  = reward / np.sum(self.betas)                        # (N,)

        temporal_diff = self.weights @ (self.gamma * succ_vec - state_vec) # (N,)
        return reward_term + temporal_diff

    def compute_delta(self, state_vec, succ_vec, reward):
        """
        Total RPE:  δ_total = Σ_i β_i δ_i  (scalar).
        """
        return float(np.dot(self.betas, self.compute_delta_feat(state_vec, succ_vec, reward)))

    # ── Two-timescale learning ─────────────────────────────────────────────────

    def learn(self, state_vec, succ_vec, reward, ret_da=True):
        """
        One-step two-timescale update.

        Order:
          1. Compute δ_feat and δ_total with OLD W and OLD β.
          2. Cache V_old = W φ(s)  (used in β gradient).
          3. Slow:  W  ← W  + α · outer(δ_feat, φ(s))
                    i.e. w_i ← w_i + α · δ_i · φ(s)   (local update)
          4. Fast:  β  ← β  + α_β · δ_total · V_old
          5. clip(β, beta_min);  optionally renormalise Σβ_i = N.

        Parameters
        ----------
        state_vec : (n_features,)
        succ_vec  : (n_features,)
        reward    : float
        ret_da    : bool — return δ_feat (the DA signal vector)

        Returns
        -------
        np.ndarray shape (N,) or None
        """
        # 1. DA signal + total error — OLD W, OLD β
        delta_feat  = self.compute_delta_feat(state_vec, succ_vec, reward)  # (N,)
        delta_total = float(np.dot(self.betas, delta_feat))

        # 2. Per-channel values with OLD W
        V_old = self.weights @ state_vec   # (N,)
        V_new = self.weights @ succ_vec       # (N,)
        # 3. Slow W update: W += α · δ_feat[:,None] · φ[None,:]
        self.weights += self.alpha * np.outer(delta_feat, state_vec)
        # self.weights += self.alpha * self.betas[:, None] * np.outer(delta_feat, state_vec)

        # 4. Fast β update: β += α_β · δ_total · V_old
        if self.alpha_beta != 0.0:
            lambda_reg = 0.1
            penalty = lambda_reg * 0.5  * np.sign(self.betas) / np.sqrt(np.abs(self.betas) + 1e-8)
            penalty = 0
            self.delta_beta = delta_total * V_old - penalty
            self.betas += self.alpha_beta * self.delta_beta
            # delta_beta = delta_total * (V_old - self.gamma * V_new) - penalty
            # delta_beta = delta_beta - np.sum(delta_beta)/self.n_values 
            # self.betas += self.alpha_beta * (delta_total * (V_old -reward /np.sum(self.betas) - self.gamma * V_new ) - penalty) +0.01
            

            np.clip(self.betas, self.beta_min, None, out=self.betas)
            if self.normalize_betas:
                self.betas *= self.n_values / self.betas.sum()

        if ret_da:
            return delta_feat   # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Self-tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from VectorRPEAgent import VectorRPEAgent

    n_states = 6
    J        = n_states   # one-hot features → J = n_states
    lr       = 0.01
    gamma    = 0.95

    def phi(s):
        v = np.zeros(J); v[s] = 1.0; return v

    # ── Test 3: N=4 channels, verify V_tot converges to V* ──────────────────
    tpe3  = TimescalePEAgent(J, n_values=4, lr=lr, gamma=gamma,
                             betas=[1.5, 0.5, 1.0, 2.0], lr_beta=10*lr)
    for _ in range(1000):
        for s in range(n_states - 1):
            r = 1.0 if s == n_states - 2 else 0.0
            tpe3.learn(phi(s), phi(s+1), r)

    # After sufficient training (e.g. 30000 passes):
    # tpe3.weights shape: (4, 6)  →  W[i, s] = V_i(s)  (one-hot features)
    # tpe3.betas   shape: (4,)

    # 1. Per-channel values at every state: shape (4, 6)
    V_channels = tpe3.weights.copy()          # W[i, s] = V_i(s)

    # 2. Weighted sum at every state: shape (6,)  ←  this should equal V_total
    V_reconstructed = tpe3.betas @ V_channels  # (4,) @ (4, 6) = (6,)

    # 3. V_total from agent.val() for comparison
    V_total = np.array([tpe3.val(phi(s)) for s in range(n_states)])

    # 4. Theoretical target: γ^d where d = distance to reward state
    reward_state = n_states - 2              # state 4 in a 6-state chain
    V_theory = np.array([
        gamma ** abs(s - reward_state) if s <= reward_state else 0.0
        for s in range(n_states)
    ])

    print("Channel values V_i(s)  [shape (N, n_states)]:")
    print(np.round(V_channels, 4))
    print()
    print(f"{'State':>6} {'V_recon':>10} {'V_total':>10} {'V_theory':>10} {'match':>8}")
    for s in range(n_states):
        match = abs(V_reconstructed[s] - V_total[s]) < 1e-10
        print(f"  s={s}   {V_reconstructed[s]:>10.4f} {V_total[s]:>10.4f} "
            f"{V_theory[s]:>10.4f}   {'✓' if match else '✗ FAIL'}")

    print()
    print("Max error |Σβ_i V_i − V_total|:", np.abs(V_reconstructed - V_total).max())
    print("Max error |V_total − V_theory|:", np.abs(V_total - V_theory).max(),
        "← nonzero until convergence")

    # 5. The key check at the reward state: Σ_i β_i V_i(reward_state) ≈ r = 1
    reward_contrib = tpe3.betas * V_channels[:, reward_state]
    print(f"\nAt reward state s={reward_state}:")
    print(f"  β:              {np.round(tpe3.betas, 4)}")
    print(f"  V_i(s):         {np.round(V_channels[:, reward_state], 4)}")
    print(f"  β_i * V_i(s):   {np.round(reward_contrib, 4)}")
    print(f"  Σ β_i V_i(s) = {reward_contrib.sum():.6f}   (target r = 1.0)")