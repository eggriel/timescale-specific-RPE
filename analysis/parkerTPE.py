"""
parkerTPE.py
============
Parker et al. (2016) two-lever reversal task, extended to run four model types
simultaneously for direct comparison:

  1. VectorRPEAgent     – feature-specific RPE (Lee et al. 2024, β_i = 1)
  2. OutcomePEAgent     – distributional / expectile RPE (Dabney et al. 2020)
  3. TimescalePEAgent_flat   – custom β model, all β_i = 1 (identical to VectorRPE)
  4. TimescalePEAgent_biased – custom β model, β biased toward the initially high-
                               reward side (left path states get β > 1, right < 1).

Task structure (identical to parkerRPE.py)
------------------------------------------
States:
  0 – start (lever presentation + ITI self-transitions)
  1 – reward outcome
  2 – omission outcome
  3, 4, 5 – left action path  (premotor → press → wait)
  6, 7, 8 – right action path

Outcome signal: external reward r ∈ {0, 1}.
(For the APE comparison, see parkerAPE.py / parkerTPE_APE.py.)

Block structure: bandit_probs flips every block_switch_interval trials.
Initial: left=0.7, right=0.1.

Saved to:  ./data/parker/TPE/DA.npz
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from VectorRPEAgent  import VectorRPEAgent
from OutcomePEAgent  import OutcomePEAgent
from TimescalePEAgent import TimescalePEAgent

np.random.seed(865612)  # same seed as parkerRPE.py for comparability

# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_phi_simple(num_states):
    def phi_simple(state):
        v = np.zeros(num_states)
        v[state] = 1.0
        return v
    return phi_simple


def softmax(vals, beta):
    e = np.exp(vals / beta)
    return e / e.sum()

# ─── Task parameters ──────────────────────────────────────────────────────────

num_bandits           = 2
bandit_probs          = np.array([0.7, 0.1])
num_trials            = 20000
block_switch_interval = 5000

left_states  = [3, 4, 5]
right_states = [left_states[-1] + 1 + i for i in range(len(left_states))]
state_paths  = [left_states, right_states]
num_states   = 3 + len(left_states) + len(right_states)   # 9
N_VALUES     = 2       # left value channel + right value channel
trial_start_probability   = 0.20
outcome_delay_probability = 0.05

phi_simple = build_phi_simple(num_states)

# ─── Agent parameters ─────────────────────────────────────────────────────────

lr          = 0.10
lr_beta     = 0.20    # β learning rate — 2× faster than w  (α_β > α_w)
gamma_rpe   = 0.95
sftmx_temp  = 0.25
num_outcome_channels = 7   # for OutcomePEAgent

# Initial β values for TimescalePEAgent_biased
# These are priors; β will be refined by learning.
# β ∝ expected reward probability associated with each state.
#   start(0)=1, reward(1)=2, omit(2)=0.5, left_path(3-5)=1.4, right_path(6-8)=0.6
# Rescaled so Σβ_i = N = 9.
_betas_raw = np.array([1.0, 2.0, 0.5,
                        1.4, 1.4, 1.4,
                        0.6, 0.6, 0.6])
_betas_biased = _betas_raw * (num_states / _betas_raw.sum())   # normalise sum → N

# ─── Agents ───────────────────────────────────────────────────────────────────

vrpe_agent    = VectorRPEAgent(num_states, lr, gamma_rpe)
outcome_agent = OutcomePEAgent(num_states, num_outcome_channels, lr, gamma_rpe)
tpe_flat      = TimescalePEAgent(num_states, N_VALUES, lr, gamma_rpe,
                                 lr_beta=lr_beta)              # β starts uniform, learned
tpe_biased    = TimescalePEAgent(num_states, N_VALUES, lr, gamma_rpe,
                                 betas=_betas_biased,
                                 lr_beta=lr_beta)              # β starts reward-biased, then learned

# ─── Storage ──────────────────────────────────────────────────────────────────

da_vrpe     = np.empty(num_trials, dtype=object)   # per-feature PE  (num_features,)
da_outcome  = np.empty(num_trials, dtype=object)   # per-channel PE  (num_channels,)
da_tpe_flat = np.empty(num_trials, dtype=object)   # per-feature PE  (num_features,)
da_tpe_bias = np.empty(num_trials, dtype=object)   # per-feature PE  (num_features,)

actions       = np.zeros(num_trials)
rewards       = np.zeros(num_trials)
reward_trials = []
states_log    = np.empty(num_trials, dtype=object)
epoch         = np.zeros(num_trials)
epoch_type    = 0

# β snapshots every snapshot_interval trials
snapshot_interval   = 500
beta_flat_history   = []   # each entry is (N_VALUES,)
beta_biased_history = []
beta_snapshot_trials= []

# ─── Simulation ───────────────────────────────────────────────────────────────

for trial in range(num_trials):

    if trial > 0 and trial % block_switch_interval == 0:
        bandit_probs  = bandit_probs[::-1]
        epoch_type    = 1 - epoch_type
    epoch[trial] = epoch_type

    if trial % snapshot_interval == 0:
        beta_flat_history.append(tpe_flat.betas.copy())
        beta_biased_history.append(tpe_biased.betas.copy())
        beta_snapshot_trials.append(trial)

    t_vrpe = []; t_outcome = []; t_tpe_flat = []; t_tpe_bias = []; t_states = []

    # ── Step 0: lever presentation + random ITI ──
    state = 0
    t_states.append(state)
    phi_s = phi_simple(state)
    phi_zero = np.zeros(num_states)

    # First DA sample (no learning, just observation at trial start)
    t_vrpe.append(vrpe_agent.compute_delta_feat(phi_zero, phi_s, 0))
    t_outcome.append(outcome_agent.compute_delta(phi_zero, phi_s, 0))
    t_tpe_flat.append(tpe_flat.compute_delta_feat(phi_zero, phi_s, 0))
    t_tpe_bias.append(tpe_biased.compute_delta_feat(phi_zero, phi_s, 0))

    while np.random.uniform() >= trial_start_probability:
        t_vrpe.append(    vrpe_agent.learn(phi_s, phi_s, 0))
        t_outcome.append( outcome_agent.learn(phi_s, phi_s, 0))
        t_tpe_flat.append(tpe_flat.learn(phi_s, phi_s, 0))
        t_tpe_bias.append(tpe_biased.learn(phi_s, phi_s, 0))
        t_states.append(state)

    # ── Step 1: choose action ──
    action = np.random.choice(2, p=softmax(bandit_probs, sftmx_temp))
    actions[trial] = action

    # ── Step 2: propagate along chosen path ──
    for path_state in state_paths[action]:
        phi_succ = phi_simple(path_state)
        t_vrpe.append(    vrpe_agent.learn(phi_s, phi_succ, 0))
        t_outcome.append( outcome_agent.learn(phi_s, phi_succ, 0))
        t_tpe_flat.append(tpe_flat.learn(phi_s, phi_succ, 0))
        t_tpe_bias.append(tpe_biased.learn(phi_s, phi_succ, 0))
        t_states.append(path_state)
        phi_s = phi_succ

    # ── Step 2.5: variable outcome delay ──
    while np.random.uniform() >= outcome_delay_probability:
        t_vrpe.append(    vrpe_agent.learn(phi_s, phi_s, 0))
        t_outcome.append( outcome_agent.learn(phi_s, phi_s, 0))
        t_tpe_flat.append(tpe_flat.learn(phi_s, phi_s, 0))
        t_tpe_bias.append(tpe_biased.learn(phi_s, phi_s, 0))
        t_states.append(t_states[-1])

    # ── Step 3: outcome ──
    reward = 1 if np.random.uniform() < bandit_probs[action] else 0
    rewards[trial] = reward
    if reward == 1:
        reward_trials.append(trial)
        outcome_state = 1
    else:
        outcome_state = 2

    phi_out = phi_simple(outcome_state)
    t_vrpe.append(    vrpe_agent.learn(phi_s, phi_out, reward))
    t_outcome.append( outcome_agent.learn(phi_s, phi_out, reward))
    t_tpe_flat.append(tpe_flat.learn(phi_s, phi_out, reward))
    t_tpe_bias.append(tpe_biased.learn(phi_s, phi_out, reward))
    t_states.append(outcome_state)

    # ── Store ──
    da_vrpe[trial]     = t_vrpe
    da_outcome[trial]  = t_outcome
    da_tpe_flat[trial] = t_tpe_flat
    da_tpe_bias[trial] = t_tpe_bias
    states_log[trial]  = t_states

# ─── Save ─────────────────────────────────────────────────────────────────────

os.makedirs('./data/parker/TPE', exist_ok=True)
np.savez('./data/parker/TPE/DA.npz',
         da_vrpe=da_vrpe,
         da_outcome=da_outcome,
         da_tpe_flat=da_tpe_flat,
         da_tpe_bias=da_tpe_bias,
         actions=actions,
         rewards=rewards,
         reward_trials=np.array(reward_trials),
         states=states_log,
         epoch=epoch,
         betas_flat=tpe_flat.betas,
         betas_biased=tpe_biased.betas,
         beta_flat_history=np.array(beta_flat_history),
         beta_biased_history=np.array(beta_biased_history),
         beta_snapshot_trials=np.array(beta_snapshot_trials),
         taus_outcome=outcome_agent.taus,
         block_switch_interval=block_switch_interval,
         num_outcome_channels=num_outcome_channels,
         gamma=gamma_rpe,
         lr=lr)

print(f"parkerTPE done.  Saved {num_trials} trials to ./data/parker/TPE/DA.npz")
print(f"  Reward trials: {len(reward_trials)} / {num_trials}")
