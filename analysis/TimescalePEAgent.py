"""
TimescalePEAgent.py
===================
Custom β-weighted feature-specific RPE agent with two-timescale learning.

An extension of the Lee et al. (2024) feature-specific RPE model where different
striatal/DA channels carry heterogeneous contribution weights β_i to the total
value signal.  Both the synaptic weights w and the channel gains β are learned,
with β updating at a faster timescale than w.

Model structure
---------------
Feature channel i receives cortical feature φ_{i,t} ∈ ℝ (one scalar component
of the full state feature vector) and maintains a scalar weight w_i.

    V_i(s)      = w_i · φ_i(s)                          (per-channel value)
    V_total(s)  = Σ_i β_i V_i(s) = (β ⊙ w) · φ(s)     (total β-weighted value)
    δ_total     = r_t + γ V_total(s') − V_total(s)       (scalar RPE on V_total)

Per-channel (feature-specific) prediction error — the DA signal:
    δ_i = r_t / (β_i · N) + w_i · (γ φ_i(s') − φ_i(s))

Consistency identity:
    Σ_i β_i δ_i = δ_total    (holds for any β_i > 0)

Two-timescale learning
----------------------
Loss:  L = ½ δ_total²

Slow update — synaptic weights w (gradient of L w.r.t. w_i):
    ∂L/∂w_i = −δ_total · β_i · φ_i(s)
    w_i  ←  w_i + α_w · β_i · δ_total · φ_i(s)

Fast update — channel gains β (gradient of L w.r.t. β_i):
    ∂L/∂β_i = −δ_total · V_i(s)   =   −δ_total · w_i · φ_i(s)
    β_i  ←  β_i + α_β · δ_total · V_i(s)

with α_β > α_w (β updates faster).

Crucially, both updates use the SAME δ_total computed with the weights and β
values from the START of the step; V_i(s) = w_i_old · φ_i(s) is saved before
w is modified, preserving the two-timescale separation.

Intuition for why β should be faster
--------------------------------------
• w encodes the *magnitude* of each channel's value prediction. This is a
  precise numerical quantity that requires many samples to estimate reliably.
• β encodes the *relevance* or *salience* of each channel — essentially which
  features matter for predicting δ_total. This is a coarser, directional signal
  (did channel i pull in the right direction?) that can be resolved quickly.
• Formally this mirrors two-timescale stochastic approximation (Borkar 1997):
  the fast β converges to a quasi-static equilibrium conditioned on slow w,
  providing a stable platform on which w can make accurate gradient steps.
• Biologically, β could correspond to a rapid gain-control or attentional
  mechanism (e.g., neuromodulatory gating of corticostriatal projections),
  while w corresponds to slower Hebbian synaptic consolidation.

β regularisation
----------------
After the β update two constraints are enforced in order:
  1. Clip:  β_i ← max(β_i, beta_min)   — prevents β from reaching 0 / going
     negative, which would invert a channel's contribution to V_total.
  2. Renormalise (optional):  β ← β · N / Σβ_i   — keeps Σβ_i = N so that
     the scale of δ_total is comparable to a standard scalar RPE and prevents
     all β_i drifting together without bound.

Special cases
-------------
• α_β = 0   : β fixed (static prior), reduces to the original model.
• β_i = 1 ∀i: δ_i = r/N + w_i(γφ_{i+1} − φ_i), identical to VectorRPEAgent.
"""

import numpy as np


class TimescalePEAgent:
    def __init__(self, num_features, lr, gamma,
                 betas=None,
                 lr_beta=None,
                 beta_min=1e-3,
                 normalize_betas=True):
        """
        Parameters
        ----------
        num_features : int
            Dimensionality of the feature vector φ(s).
        lr : float
            Learning rate for w (slow timescale), α_w.
        gamma : float
            Temporal discount factor γ.
        betas : array-like of shape (num_features,) or None
            Initial channel gain vector.  None → all 1s (= VectorRPEAgent).
            Any strictly positive values are valid; the model normalises
            internally if normalize_betas=True.
        lr_beta : float or None
            Learning rate for β (fast timescale), α_β.
            None → 2 × lr  (twice as fast as w by default).
            Set to 0 to disable β learning (fixed gains).
        beta_min : float
            Hard lower bound on β_i after each update.  Prevents β from
            crossing zero, which would invert a channel's sign in V_total.
            Default: 1e-3.
        normalize_betas : bool
            If True (default), renormalise β after each update so Σβ_i = N.
            This keeps the scale of δ_total comparable to a standard RPE and
            prevents unbounded growth.
        """
        self.num_features    = num_features
        self.alpha           = lr
        self.alpha_beta      = (2.0 * lr) if lr_beta is None else float(lr_beta)
        self.gamma           = gamma
        self.beta_min        = beta_min
        self.normalize_betas = normalize_betas

        self.weights = np.zeros(num_features)

        if betas is None:
            self.betas = np.ones(num_features)
        else:
            self.betas = np.asarray(betas, dtype=float).copy()
            if np.any(self.betas <= 0):
                raise ValueError("All initial β_i must be strictly positive.")

    # ------------------------------------------------------------------
    # Value computation
    # ------------------------------------------------------------------

    def val(self, state_vec):
        """
        Total β-weighted value:  V_total(s) = (β ⊙ w) · φ(s).

        Parameters
        ----------
        state_vec : np.ndarray, shape (num_features,)

        Returns
        -------
        float
        """
        return float(np.dot(self.betas * self.weights, state_vec))

    def val_per_channel(self, state_vec):
        """
        Per-channel values:  V_i(s) = w_i · φ_i(s).

        Returns
        -------
        np.ndarray, shape (num_features,)
        """
        return self.weights * state_vec

    # ------------------------------------------------------------------
    # Prediction error computation
    # ------------------------------------------------------------------

    def compute_delta_feat(self, state_vec, succ_vec, reward):
        """
        Per-channel prediction errors — the DA signal:

            δ_i = r / (β_i · N) + w_i · (γ · φ_i(s') − φ_i(s))

        Parameters
        ----------
        state_vec : np.ndarray, shape (num_features,)
        succ_vec  : np.ndarray, shape (num_features,)
        reward    : float

        Returns
        -------
        np.ndarray, shape (num_features,)
        """
        N             = self.num_features
        reward_term   = reward / (self.betas * N)
        temporal_diff = self.weights * (self.gamma * succ_vec - state_vec)
        return reward_term + temporal_diff

    def compute_delta(self, state_vec, succ_vec, reward):
        """
        Total scalar RPE:  δ_total = Σ_i β_i δ_i = r + γ V_total(s') − V_total(s).

        Returns
        -------
        float
        """
        return float(np.dot(self.betas,
                            self.compute_delta_feat(state_vec, succ_vec, reward)))

    # ------------------------------------------------------------------
    # Two-timescale learning
    # ------------------------------------------------------------------

    def learn(self, state_vec, succ_vec, reward, ret_da=True):
        """
        Single-step two-timescale update.

        Order of operations (preserving two-timescale separation):
          1. Compute δ_feat and δ_total with CURRENT (old) w and β.
          2. Cache V_i = w_i · φ_i(s) with OLD w — used in β gradient.
          3. Slow update:  w  ←  w  +  α_w · β · δ_total · φ(s)
          4. Fast update:  β  ←  β  +  α_β · δ_total · V_old
          5. Clip β to beta_min; renormalise if normalize_betas=True.

        Parameters
        ----------
        state_vec : np.ndarray, shape (num_features,)
        succ_vec  : np.ndarray, shape (num_features,)
        reward    : float
        ret_da    : bool
            Return per-channel δ_i (DA signal vector) if True.

        Returns
        -------
        np.ndarray, shape (num_features,) or None
        """
        # 1. DA signal + total PE — computed with OLD w and OLD β
        delta_feat  = self.compute_delta_feat(state_vec, succ_vec, reward)
        delta_total = float(np.dot(self.betas, delta_feat))

        # 2. Per-channel value with OLD w  (used for β gradient)
        V_old = self.weights * state_vec   # V_i = w_i · φ_i(s)

        # 3. Slow w update:  w += α_w · β · δ_total · φ(s)
        self.weights += self.alpha * self.betas * delta_total * state_vec

        # 4. Fast β update:  β += α_β · δ_total · V_old
        #    ∂L/∂β_i = −δ_total · V_i  →  gradient-descent step adds α_β·δ_total·V_i
        if self.alpha_beta != 0.0:
            self.betas += self.alpha_beta * delta_total * V_old

            # 5. Regularise β
            np.clip(self.betas, self.beta_min, None, out=self.betas)
            if self.normalize_betas:
                self.betas *= self.num_features / self.betas.sum()

        if ret_da:
            return delta_feat   # shape (num_features,)


# ---------------------------------------------------------------------------
# Stand-alone tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from VectorRPEAgent import VectorRPEAgent
    import sys

    num_states = 6
    lr         = 0.05
    gamma      = 0.95

    def phi(s):
        v = np.zeros(num_states)
        v[s] = 1.0
        return v

    # ── Test 1: flat β (no update) matches VectorRPEAgent ──────────────
    vrpe      = VectorRPEAgent(num_states, lr, gamma)
    tpe_fixed = TimescalePEAgent(num_states, lr, gamma, lr_beta=0)  # β fixed

    for _ in range(20000):
        for s in range(num_states - 1):
            r = 1.0 if s == num_states - 2 else 0.0
            vrpe.learn(phi(s), phi(s + 1), r)
            tpe_fixed.learn(phi(s), phi(s + 1), r)

    print("── Test 1: fixed β=1 vs VectorRPEAgent ──")
    print("VectorRPE w: ", np.round(vrpe.weights, 4))
    print("TPE fixed w: ", np.round(tpe_fixed.weights, 4))
    print("Match:       ", np.allclose(vrpe.weights, tpe_fixed.weights, atol=1e-6))

    # ── Test 2: learned β at 2× rate ───────────────────────────────────
    tpe_learn = TimescalePEAgent(num_states, lr, gamma,
                                 lr_beta=2*lr,       # 2× faster β
                                 normalize_betas=True)

    beta_history = [tpe_learn.betas.copy()]
    for _ in range(20000):
        for s in range(num_states - 1):
            r = 1.0 if s == num_states - 2 else 0.0
            tpe_learn.learn(phi(s), phi(s + 1), r)
        if _ % 4000 == 0:
            beta_history.append(tpe_learn.betas.copy())

    print("\n── Test 2: learned β (α_β = 2α_w) ──")
    print("Final β: ", np.round(tpe_learn.betas, 4))
    print("Σβ/N = 1?", np.isclose(tpe_learn.betas.mean(), 1.0))
    print("V_total(s=0):", round(tpe_learn.val(phi(0)), 4),
          "  (VectorRPE: {:.4f})".format(vrpe.weights[0]))

    # β should be larger for states closer to reward
    print("β monotonically increasing toward reward state?",
          all(tpe_learn.betas[i] <= tpe_learn.betas[i+1]
              for i in range(num_states - 2)))
