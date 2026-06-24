"""
GridworldPE.py
==============
7×7 grid-world navigation task with block structure.
Trains five model types in parallel and saves rich data for visualisation.

Models
------
1. ScalarRPE      — classic tabular TD (ScalarRPEAgent)
2. OutcomePE      — distributional / expectile (OutcomePEAgent, N=7 channels)
3. VectorRPE      — feature-specific RPE (VectorRPEAgent, RBF features)
4. VectorAPE      — feature-specific APE (4 directional agents + RPE nav agent)
5. TimescaleRPE   — custom β-weighted RPE (TimescalePEAgent, learned β)

Task
----
Agent navigates to a goal position (+1 reward, episode ends).
Goal location changes every block.  Block schedule cycles through 4 inner
corners so the agent must adapt repeatedly.

Features
--------
Gaussian RBF centred at every grid cell (49 features for a 7×7 grid).
phi_all[s, i] = exp(-||pos(s) - center(i)||² / 2σ²)
Precomputed for all states as a lookup table.

Saved to
--------
./data/gridworld_results.npz
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ScalarRPEAgent   import ScalarRPEAgent
from VectorRPEAgent   import VectorRPEAgent
from OutcomePEAgent   import OutcomePEAgent
from TimescalePEAgent import TimescalePEAgent

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Hyper-parameters
# ═══════════════════════════════════════════════════════════════════════════════
H, W          = 7, 7
N_STATES      = H * W          # 49
N_FEATURES    = H * W          # one RBF per cell
RBF_SIGMA     = 1.5
N_OUT_CH      = 7              # expectile channels for OutcomePE
LR_W          = 0.05
LR_BETA       = 0.10           # β lr — 2× LR_W
GAMMA_RPE     = 0.95
GAMMA_APE     = 0.50           # shorter horizon for action-frequency prediction
SOFTMAX_TEMP  = 0.5
EPS_PER_BLOCK = 100
N_BLOCKS      = 8
MAX_STEPS     = 60
CKPT_INTERVAL = 25             # save maps every N episodes

# Goal schedule: 4 inner corners, visited twice
GOALS = [
    (1, 1), (5, 5), (1, 5), (5, 1),
    (1, 1), (5, 5), (1, 5), (5, 1),
]
assert len(GOALS) == N_BLOCKS

ACTION_DELTAS = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])  # N S E W
ACTION_NAMES  = ['North', 'South', 'East', 'West']

# ═══════════════════════════════════════════════════════════════════════════════
# Environment helpers
# ═══════════════════════════════════════════════════════════════════════════════

def pos_to_idx(r, c):
    return int(r * W + c)

def idx_to_pos(idx):
    return idx // W, idx % W

def clip_pos(r, c):
    return max(0, min(H-1, r)), max(0, min(W-1, c))

def get_successors(r, c):
    """All (action, next_r, next_c) from (r,c)."""
    succs = []
    for a, (dr, dc) in enumerate(ACTION_DELTAS):
        nr, nc = clip_pos(r + dr, c + dc)
        succs.append((a, nr, nc))
    return succs


class GridWorld:
    def __init__(self):
        self.goal = None
        self.r = self.c = 0
        self._steps = 0

    def reset(self, goal):
        self.goal = goal
        while True:
            r = np.random.randint(H)
            c = np.random.randint(W)
            if (r, c) != goal:
                self.r, self.c = r, c
                break
        self._steps = 0
        return pos_to_idx(self.r, self.c)

    def step(self, action):
        dr, dc = ACTION_DELTAS[action]
        self.r, self.c = clip_pos(self.r + dr, self.c + dc)
        self._steps += 1
        at_goal = (self.r, self.c) == self.goal
        reward   = 1.0 if at_goal else 0.0
        done     = at_goal or (self._steps >= MAX_STEPS)
        return pos_to_idx(self.r, self.c), reward, done

# ═══════════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════════

def build_phi_all():
    """Precomputed RBF feature table: phi_all[state_idx] = (N_FEATURES,) vector."""
    centers = np.array([(r, c) for r in range(H) for c in range(W)], dtype=float)
    phi = np.zeros((N_STATES, N_FEATURES))
    for s in range(N_STATES):
        r, c   = idx_to_pos(s)
        dists2 = np.sum((centers - [r, c]) ** 2, axis=1)
        phi[s] = np.exp(-dists2 / (2 * RBF_SIGMA**2))
    return phi

PHI = build_phi_all()   # shape (49, 49) — global lookup

# ═══════════════════════════════════════════════════════════════════════════════
# Policy
# ═══════════════════════════════════════════════════════════════════════════════

def _agent_val(agent, s_idx):
    """Scalar value estimate for state s_idx, regardless of agent type."""
    if isinstance(agent, ScalarRPEAgent):
        return float(agent.val(s_idx))
    v = np.asarray(agent.val(PHI[s_idx]))
    return float(np.mean(v)) if v.ndim > 0 and v.shape != () else float(v)


def choose_action(agent, r, c, temp=SOFTMAX_TEMP):
    """Softmax over successor-state values."""
    succs  = get_successors(r, c)
    vals   = np.array([_agent_val(agent, pos_to_idx(nr, nc)) for _, nr, nc in succs])
    e      = np.exp((vals - vals.max()) / max(temp, 1e-9))
    probs  = e / e.sum()
    a_idx  = np.random.choice(len(succs), p=probs)
    return succs[a_idx][0]   # action integer

# ═══════════════════════════════════════════════════════════════════════════════
# Unified episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(env, goal, learn_agent, policy_agent=None,
                is_scalar=False, ape_direction=None):
    """
    Run one episode.

    Parameters
    ----------
    learn_agent   : agent to update (may differ from policy agent for APE)
    policy_agent  : if not None, used for action selection
    is_scalar     : True for ScalarRPEAgent
    ape_direction : int 0-3 → action indicator reward; None → environment reward

    Returns
    -------
    steps        : int
    success      : float  (1.0 if goal reached)
    traj_s       : list of state indices visited
    da_per_step  : list of (state_idx, float |δ_scalar|)
    """
    s = env.reset(goal)
    nav = policy_agent if policy_agent is not None else learn_agent
    traj  = [s]
    da_ps = []
    done  = False

    while not done:
        r, c   = env.r, env.c
        action = choose_action(nav, r, c)
        ns, reward, done = env.step(action)

        lr_rew = float(action == ape_direction) if ape_direction is not None else reward

        if is_scalar:
            da = learn_agent.learn(s, ns, lr_rew)
            da_ps.append((s, abs(float(da))))
        else:
            da = learn_agent.learn(PHI[s], PHI[ns], lr_rew)
            da_ps.append((s, float(np.mean(np.abs(da)))))

        s = ns
        traj.append(s)

    return len(traj)-1, float((env.r, env.c) == goal), traj, da_ps

# ═══════════════════════════════════════════════════════════════════════════════
# Map computation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def value_map(agent):
    """H×W array of V(s) for every grid state."""
    vm = np.zeros(N_STATES)
    for s in range(N_STATES):
        vm[s] = _agent_val(agent, s)
    return vm.reshape(H, W)


def da_map_from_log(da_log):
    """H×W mean-|δ| map from list of (state_idx, |δ|) pairs."""
    acc = np.zeros(N_STATES)
    cnt = np.zeros(N_STATES)
    for s, d in da_log:
        acc[s] += d; cnt[s] += 1
    return (acc / np.maximum(cnt, 1)).reshape(H, W)


def beta_map(agent):
    """H×W map of β_i values (TimescalePEAgent only)."""
    return agent.betas.reshape(H, W)


def channel_val_maps(agent):
    """(N_channels, H, W) per-channel value maps for OutcomePEAgent."""
    maps = np.zeros((agent.num_channels, N_STATES))
    for s in range(N_STATES):
        maps[:, s] = agent.val(PHI[s])
    return maps.reshape(agent.num_channels, H, W)


def feat_da_map(agent, s_idx, ns_idx, reward):
    """(N_FEATURES, H, W) per-feature DA map from a single transition (illustrative)."""
    df = agent.compute_delta_feat(PHI[s_idx], PHI[ns_idx], reward)
    return df.reshape(H, W)   # one channel's spatial tuning

# ═══════════════════════════════════════════════════════════════════════════════
# Agents
# ═══════════════════════════════════════════════════════════════════════════════

agents = {
    'ScalarRPE':   ScalarRPEAgent(N_STATES,    lr=LR_W,   gamma=GAMMA_RPE),
    'OutcomePE':   OutcomePEAgent(N_FEATURES,  N_OUT_CH,  lr=LR_W, gamma=GAMMA_RPE),
    'VectorRPE':   VectorRPEAgent(N_FEATURES,  lr=LR_W,   gamma=GAMMA_RPE),
    'TimescaleRPE':TimescalePEAgent(N_FEATURES, lr=LR_W,  gamma=GAMMA_RPE,
                                    lr_beta=LR_BETA, normalize_betas=True),
}

# APE: nav agent + 4 directional observers
ape_nav  = VectorRPEAgent(N_FEATURES, lr=LR_W,  gamma=GAMMA_RPE)
ape_dirs = [VectorRPEAgent(N_FEATURES, lr=LR_W, gamma=GAMMA_APE)
            for _ in range(4)]

# ═══════════════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════════════

TOTAL_EPS = EPS_PER_BLOCK * N_BLOCKS
N_CKPTS   = TOTAL_EPS // CKPT_INTERVAL + 1

store = {name: {
    'steps':     np.zeros(TOTAL_EPS),
    'success':   np.zeros(TOTAL_EPS),
    'td_loss':   np.zeros(TOTAL_EPS),
    'val_maps':  np.zeros((N_CKPTS, H, W)),
    'da_maps':   np.zeros((N_CKPTS, H, W)),
} for name in list(agents) + ['VectorAPE']}

# Extra per-model stores
store['OutcomePE']['chan_val_maps']  = np.zeros((N_CKPTS, N_OUT_CH, H, W))
store['TimescaleRPE']['beta_maps']  = np.zeros((N_CKPTS, H, W))
store['VectorAPE']['ape_da_maps']   = np.zeros((N_CKPTS, 4, H, W))
store['VectorAPE']['nav_val_maps']  = np.zeros((N_CKPTS, H, W))

block_goals = np.zeros((N_BLOCKS, 2), dtype=int)
block_eps   = np.zeros(N_BLOCKS, dtype=int)
checkpoints = []

# DA accumulators (reset at each checkpoint)
da_acc = {name: [] for name in list(agents) + ['VectorAPE']}
ape_da_acc = [[] for _ in range(4)]   # per direction

env = GridWorld()

# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

ckpt_idx = 0

def save_checkpoint(ep_idx, goal):
    global ckpt_idx
    if ckpt_idx >= N_CKPTS:
        return
    checkpoints.append(ep_idx)

    # ── Scalar / Vector / Timescale / Outcome value + DA maps ──
    for name, ag in agents.items():
        store[name]['val_maps'][ckpt_idx]  = value_map(ag)
        store[name]['da_maps'][ckpt_idx]   = da_map_from_log(da_acc[name])
        da_acc[name].clear()

    store['OutcomePE']['chan_val_maps'][ckpt_idx]  = channel_val_maps(agents['OutcomePE'])
    store['TimescaleRPE']['beta_maps'][ckpt_idx]   = beta_map(agents['TimescaleRPE'])

    # ── APE ──
    store['VectorAPE']['val_maps'][ckpt_idx]     = value_map(ape_nav)
    store['VectorAPE']['nav_val_maps'][ckpt_idx] = value_map(ape_nav)
    store['VectorAPE']['da_maps'][ckpt_idx]      = da_map_from_log(da_acc['VectorAPE'])
    da_acc['VectorAPE'].clear()
    for d in range(4):
        store['VectorAPE']['ape_da_maps'][ckpt_idx, d] = da_map_from_log(ape_da_acc[d])
        ape_da_acc[d].clear()

    ckpt_idx += 1


print(f"Training: {N_BLOCKS} blocks × {EPS_PER_BLOCK} episodes = {TOTAL_EPS} episodes")
print(f"Grid: {H}×{W}  |  Features: {N_FEATURES} RBFs (σ={RBF_SIGMA})")
print(f"Models: ScalarRPE, OutcomePE ({N_OUT_CH}ch), VectorRPE, VectorAPE (4-dir), TimescaleRPE")

for blk in range(N_BLOCKS):
    goal = GOALS[blk]
    block_goals[blk] = goal
    block_eps[blk]   = blk * EPS_PER_BLOCK

    for ep_in_blk in range(EPS_PER_BLOCK):
        ep = blk * EPS_PER_BLOCK + ep_in_blk

        if ep % CKPT_INTERVAL == 0:
            save_checkpoint(ep, goal)

        # ── ScalarRPE ──────────────────────────────────────────────────────
        steps, succ, _, dp = run_episode(env, goal, agents['ScalarRPE'],
                                         is_scalar=True)
        store['ScalarRPE']['steps'][ep]   = steps
        store['ScalarRPE']['success'][ep] = succ
        store['ScalarRPE']['td_loss'][ep] = np.mean([d for _, d in dp]) if dp else 0
        da_acc['ScalarRPE'].extend(dp)

        # ── OutcomePE ──────────────────────────────────────────────────────
        steps, succ, _, dp = run_episode(env, goal, agents['OutcomePE'])
        store['OutcomePE']['steps'][ep]   = steps
        store['OutcomePE']['success'][ep] = succ
        store['OutcomePE']['td_loss'][ep] = np.mean([d for _, d in dp]) if dp else 0
        da_acc['OutcomePE'].extend(dp)

        # ── VectorRPE ──────────────────────────────────────────────────────
        steps, succ, _, dp = run_episode(env, goal, agents['VectorRPE'])
        store['VectorRPE']['steps'][ep]   = steps
        store['VectorRPE']['success'][ep] = succ
        store['VectorRPE']['td_loss'][ep] = np.mean([d for _, d in dp]) if dp else 0
        da_acc['VectorRPE'].extend(dp)

        # ── VectorAPE ──────────────────────────────────────────────────────
        # Nav agent learns from reward; 4 APE agents learn from action indicators
        s = env.reset(goal)
        nav_dp = []; ape_dp = [[] for _ in range(4)]
        nav_steps = 0; nav_succ = 0.0; done = False

        while not done:
            r, c   = env.r, env.c
            action = choose_action(ape_nav, r, c)
            ns, reward, done = env.step(action)

            # nav update (reward-based)
            dv = ape_nav.learn(PHI[s], PHI[ns], reward)
            nav_dp.append((s, float(np.mean(np.abs(dv)))))

            # APE updates (action-indicator reward)
            for d_idx, ape_ag in enumerate(ape_dirs):
                ape_r = float(action == d_idx)
                da_v  = ape_ag.learn(PHI[s], PHI[ns], ape_r)
                ape_dp[d_idx].append((s, float(np.mean(np.abs(da_v)))))

            s = ns
            nav_steps += 1
            if (env.r, env.c) == goal:
                nav_succ = 1.0

        store['VectorAPE']['steps'][ep]   = nav_steps
        store['VectorAPE']['success'][ep] = nav_succ
        store['VectorAPE']['td_loss'][ep] = np.mean([d for _, d in nav_dp]) if nav_dp else 0
        da_acc['VectorAPE'].extend(nav_dp)
        for d_idx in range(4):
            ape_da_acc[d_idx].extend(ape_dp[d_idx])

        # ── TimescaleRPE ───────────────────────────────────────────────────
        steps, succ, _, dp = run_episode(env, goal, agents['TimescaleRPE'])
        store['TimescaleRPE']['steps'][ep]   = steps
        store['TimescaleRPE']['success'][ep] = succ
        store['TimescaleRPE']['td_loss'][ep] = np.mean([d for _, d in dp]) if dp else 0
        da_acc['TimescaleRPE'].extend(dp)

    print(f"  Block {blk+1}/{N_BLOCKS} (goal={goal})  "
          f"final success: "
          f"Scalar={np.mean(store['ScalarRPE']['success'][blk*EPS_PER_BLOCK:(blk+1)*EPS_PER_BLOCK]):.2f} "
          f"VecRPE={np.mean(store['VectorRPE']['success'][blk*EPS_PER_BLOCK:(blk+1)*EPS_PER_BLOCK]):.2f} "
          f"TscRPE={np.mean(store['TimescaleRPE']['success'][blk*EPS_PER_BLOCK:(blk+1)*EPS_PER_BLOCK]):.2f}")

# Final checkpoint
save_checkpoint(TOTAL_EPS, GOALS[-1])

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════


save_dict = dict(
    block_goals=block_goals,
    block_eps=block_eps,
    checkpoints=np.array(checkpoints),
    H=H, W=W, N_FEATURES=N_FEATURES, N_OUT_CH=N_OUT_CH,
    EPS_PER_BLOCK=EPS_PER_BLOCK, N_BLOCKS=N_BLOCKS,
    GAMMA_RPE=GAMMA_RPE, GAMMA_APE=GAMMA_APE,
    LR_W=LR_W, LR_BETA=LR_BETA, RBF_SIGMA=RBF_SIGMA,
    phi_all=PHI,
)

for name, d in store.items():
    for k, v in d.items():
        save_dict[f'{name}__{k}'] = v

np.savez('./data/gridworld/results.npz', **save_dict)
print(f"\nSaved to ./data/gridworld/results.npz")
