"""
jincostaTPE.py
==============
Jin & Costa (2010) sequential lever-press task, extended to run four model types
simultaneously in ACTION PREDICTION ERROR (APE) mode:

  1. VectorRPEAgent     – feature-specific APE (Lee et al. 2024)
  2. OutcomePEAgent     – distributional APE  (expectile over action frequency)
  3. TimescalePEAgent_flat   – custom β model, all β_i = 1
  4. TimescalePEAgent_timescale – β grows with press index (later presses get
                                  higher β, capturing a temporal 'chunking' prior)

Task structure (identical to jincosta.py)
-----------------------------------------
Agent must press a lever 3 times to obtain reward.  At each press the agent
enters a premotor state (musc) before committing to the press.

Features: 'chunked graded' Gaussian tuning curves over press-count states,
replicated num_copies=3 times (total dim = 21).

APE reward signal: I[action == preferred_action] (1 if preferred side pressed, else 0)
This makes the agents learn to predict action sequences, not reward outcomes.

Timescale β
-----------
The 'timescale' interpretation: features representing later presses in the
sequence (closer to sequence completion) are given higher β.  This reflects
the idea that neurons closer in time to action execution carry more weight.

β pattern: three copies of base features get linearly increasing β:
  copy 1 (press 1 / start): β_scale = 0.5
  copy 2 (press 2):          β_scale = 1.0
  copy 3 (press 3):          β_scale = 1.5
Rescaled so Σβ_i = num_features.

Saved to: ./data/jincosta/TPE/DA.npz
"""

import sys
import os
import math
import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(__file__))
from VectorRPEAgent   import VectorRPEAgent
from OutcomePEAgent   import OutcomePEAgent
from TimescalePEAgent import TimescalePEAgent

np.random.seed(865612)

# ─── Feature builders ─────────────────────────────────────────────────────────

def build_phi_chunked_graded(press_thresh, num_copies, is_weighted,
                             prop_first=0, scale=1):
    """Identical to jincosta.py – Gaussian-tuned chunked features."""
    def phi_chunked_graded(press_count, is_musc):
        num_neurons = 1 + 2 * press_thresh
        phi_base = np.zeros((1, num_neurons))
        for i in range(num_neurons):
            if press_count == 0:
                mean = press_count
            elif not is_musc:
                mean = press_count * 2 - 1
            else:
                mean = press_count * 2
            phi_base[0, i] = stats.norm(loc=mean, scale=scale).pdf(i)

        if not is_weighted:
            phi = np.tile(phi_base, (num_copies, 1))
        else:
            phi = np.zeros((num_copies, num_neurons))
            for i in range(num_copies):
                if i == 0:
                    phi[i, :] = phi_base
                else:
                    first_n = math.ceil(prop_first * num_neurons)
                    phi[i, :first_n] = phi_base[0, 1]
                    phi[i, first_n:] = phi_base[0, -2]
        return phi.flatten()
    return phi_chunked_graded


def scal_state(press_count, is_musc):
    """Scalar state index for the scalar APE model."""
    return 2 * press_count if is_musc else 2 * press_count - 1


def softmax(vals, beta):
    e = np.exp(vals / beta)
    return e / e.sum()

# ─── Task parameters ──────────────────────────────────────────────────────────

num_bandits           = 2
bandit_rewards        = np.array([1, 0])   # reward for each arm (flips at block switch)
press_thresh          = 3
num_trials            = 5000
block_switch_interval = 10000   # effectively no switch within 5000 trials
initiation_prob       = 0.20
seq_press_prob        = 0.90

# Agent hypers
lr          = 0.05
lr_beta     = 0.10    # β learning rate — 2× faster than w
gamma       = 0.50      # harsher discount for APE (action-level horizon)
sftmx_temp  = 0.25
prop_first  = 0.60
is_chunked  = True
tuning_scale= 0.30
num_copies  = 3
num_states  = 1 + 2 * press_thresh          # 7
chunked_features = num_states * num_copies  # 21
N_VALUES         = 2                         # left value channel + right value channel
num_outcome_channels = 7

phi_cg = build_phi_chunked_graded(press_thresh, num_copies, is_chunked,
                                   prop_first, scale=tuning_scale)

# Timescale betas: copy k gets β_scale = 0.5 * k  (k = 1,2,3)
# So features 0-6 → β=0.5, 7-13 → β=1.0, 14-20 → β=1.5
_beta_scales = np.repeat([0.5, 1.0, 1.5], num_states)        # shape (21,)
_betas_ts    = _beta_scales * (chunked_features / _beta_scales.sum())

# ─── Agents: left-preferring and right-preferring pairs ───────────────────────
#  Each model type has two instances: one that treats left press as "reward",
#  one that treats right press as "reward".  This mirrors jincosta.py's design.

# β for timescale model: N_VALUES=2 channels; first channel gets lower β (earlier timescale)
_betas_ts_vals = np.array([0.6, 1.4]) * (N_VALUES / np.array([0.6, 1.4]).sum())

agents = {
    'vrpe':      (VectorRPEAgent(chunked_features, lr, gamma),
                  VectorRPEAgent(chunked_features, lr, gamma)),
    'outcome':   (OutcomePEAgent(chunked_features, num_outcome_channels, lr, gamma),
                  OutcomePEAgent(chunked_features, num_outcome_channels, lr, gamma)),
    'tpe_flat':  (TimescalePEAgent(chunked_features, N_VALUES, lr, gamma, lr_beta=lr_beta),
                  TimescalePEAgent(chunked_features, N_VALUES, lr, gamma, lr_beta=lr_beta)),
    'tpe_ts':    (TimescalePEAgent(chunked_features, N_VALUES, lr, gamma, betas=_betas_ts_vals, lr_beta=lr_beta),
                  TimescalePEAgent(chunked_features, N_VALUES, lr, gamma, betas=_betas_ts_vals, lr_beta=lr_beta)),
}
# agents[key] = (left_agent, right_agent)

# ─── Storage ──────────────────────────────────────────────────────────────────

def empty_store():
    return (np.empty(num_trials, dtype=object),
            np.empty(num_trials, dtype=object))

da_stores = {k: empty_store() for k in agents}   # {key: (da_left, da_right)}

states_log  = np.empty(num_trials, dtype=object)
actions_log = np.empty(num_trials, dtype=object)
epoch       = np.zeros(num_trials)
trial_type  = 0

# ─── Helper: unified DA signal dispatch ───────────────────────────────────────
# VectorRPEAgent / TimescalePEAgent expose compute_delta_feat → (F,)
# OutcomePEAgent exposes compute_delta → (N_channels,)
# Both return the "DA signal vector" for a given transition.

def _da_signal(agent, state_vec, succ_vec, reward):
    """Return the DA signal (per-feature or per-channel PE vector) without updating weights."""
    if hasattr(agent, 'compute_delta_feat'):
        return agent.compute_delta_feat(state_vec, succ_vec, reward)
    else:
        return agent.compute_delta(state_vec, succ_vec, reward)


# ─── Simulation ───────────────────────────────────────────────────────────────

press_counts = np.zeros(num_bandits, dtype=int)
musc_init    = False

for trial in range(num_trials):

    if trial % block_switch_interval == 0 and trial > 0:
        bandit_rewards = bandit_rewards[::-1]
        trial_type = 1 - trial_type
    epoch[trial] = trial_type

    press_counts[:] = 0
    musc_init = False

    # Per-trial lists: {key: [left_da_steps, right_da_steps]}
    t_da = {k: ([], []) for k in agents}
    t_states  = []
    t_actions = []

    # ── Lever presentation state ──
    l_phi = phi_cg(press_counts[0], musc_init)
    r_phi = phi_cg(press_counts[1], musc_init)
    phi_zero = np.zeros(chunked_features)

    for key, (lag, rag) in agents.items():
        t_da[key][0].append(_da_signal(lag, phi_zero, l_phi, 0))
        t_da[key][1].append(_da_signal(rag, phi_zero, r_phi, 0))
    t_states.append(press_counts.tolist())

    # ── ITI self-transitions ──
    while np.random.uniform() >= initiation_prob:
        for key, (lag, rag) in agents.items():
            t_da[key][0].append(lag.learn(l_phi, l_phi, 0))
            t_da[key][1].append(rag.learn(r_phi, r_phi, 0))
        t_states.append(press_counts.tolist())

    # ── Press sequence ──
    while np.max(press_counts) < press_thresh or musc_init:
        action = -1

        if musc_init:
            action    = next_action
            next_l    = phi_cg(press_counts[0], musc_init)
            next_r    = phi_cg(press_counts[1], musc_init)
            musc_init = False

        elif np.random.uniform() >= seq_press_prob:   # wait
            next_l = l_phi
            next_r = r_phi

        else:   # choose and execute a press
            next_action = np.random.choice(2, p=softmax(bandit_rewards, sftmx_temp))
            press_counts[next_action] += 1
            next_l    = phi_cg(press_counts[0], musc_init)
            next_r    = phi_cg(press_counts[1], musc_init)
            musc_init = True

        l_reward = int(action == 0)
        r_reward = int(action == 1)

        for key, (lag, rag) in agents.items():
            t_da[key][0].append(lag.learn(l_phi, next_l, l_reward))
            t_da[key][1].append(rag.learn(r_phi, next_r, r_reward))

        t_states.append(press_counts.tolist())
        t_actions.append(action)
        l_phi = next_l
        r_phi = next_r

    # ── Store ──
    for key in agents:
        da_stores[key][0][trial] = t_da[key][0]
        da_stores[key][1][trial] = t_da[key][1]
    states_log[trial]  = t_states
    actions_log[trial] = t_actions

# ─── Save ─────────────────────────────────────────────────────────────────────

os.makedirs('./data/jincosta/TPE', exist_ok=True)
np.savez('./data/jincosta/TPE/DA.npz',
         da_vrpe_L=da_stores['vrpe'][0],
         da_vrpe_R=da_stores['vrpe'][1],
         da_outcome_L=da_stores['outcome'][0],
         da_outcome_R=da_stores['outcome'][1],
         da_tpe_flat_L=da_stores['tpe_flat'][0],
         da_tpe_flat_R=da_stores['tpe_flat'][1],
         da_tpe_ts_L=da_stores['tpe_ts'][0],
         da_tpe_ts_R=da_stores['tpe_ts'][1],
         states=states_log,
         actions=actions_log,
         epoch=epoch,
         taus_outcome=agents['outcome'][0].taus,
         betas_ts=_betas_ts,
         gamma=gamma,
         press_thresh=press_thresh,
         num_outcome_channels=num_outcome_channels,
         chunked_features=chunked_features)

print(f"jincostaTPE done.  Saved {num_trials} trials to ./data/jincosta/TPE/DA.npz")
