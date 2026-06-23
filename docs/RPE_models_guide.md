# Dopamine Prediction Error Models: A Conceptual and Mathematical Guide

## Overview

This guide covers five major computational models of dopaminergic prediction errors, a custom extension of the feature-specific model, and a grid-world task design for benchmarking them. All models share a common foundation in reinforcement learning (RL) temporal difference (TD) methods and are united by the hypothesis that phasic dopamine (DA) signals encode some form of prediction error. The models diverge in **what** is being predicted, **how** state is represented, and **which** units receive which signals.

**Key taxonomic split**: Models can be divided into:
- **Outcome-specific** (distributional RL, APE, SR): replicate the scalar RPE circuit in parallel, each predicting a different target outcome (reward quantile, action, future state). Heterogeneity arises *between* projection-defined populations.
- **Feature-specific**: a single circuit where the state representation is distributed across channels. Heterogeneity arises *within* a projection-defined population from the non-uniform convergence of cortical features onto striatum/DA.

---

## 0. Shared Framework and Notation

### The RL Problem

An agent interacts with an environment in discrete steps. At time $t$:
- Observes state $s_t \in \mathcal{S}$
- Takes action $a_t \in \mathcal{A}$
- Receives reward $r_t \in \mathbb{R}$
- Transitions to state $s_{t+1}$

The **return** (discounted cumulative reward) from time $t$ is:
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

where $\gamma \in [0,1)$ is the discount factor.

### Common Notation Table

| Symbol | Meaning |
|--------|---------|
| $s_t$ | State at time $t$ |
| $r_t$ | Reward at time $t$ |
| $\gamma$ | Temporal discount factor |
| $\phi_t$, $\vec{\phi}_t$ | Feature vector at time $t$ (scalar or vector) |
| $w$, $\vec{w}$ | Learned weight vector |
| $V(s)$ | Value function for state $s$ |
| $\delta_t$ | Prediction error (TD error) at time $t$ |
| $\alpha$, $\eta$ | Learning rate |
| $N$ | Number of channels/units |
| $i$ | Channel/unit index |
| $\beta_i$ | Channel weight (custom model) |

### The Dopamine–RPE Connection

Schultz, Dayan & Montague (1997) established the classical mapping between phasic DA signals and the TD error $\delta_t$. Three canonical signatures follow directly from the math:

1. **Before learning**: Reward is unexpected → $r_t > 0$ with $V(s_t) \approx 0$ → $\delta_t > 0$ → DA burst at reward time.
2. **After learning with CS**: CS predicts reward → $V(\text{CS}) \approx r$ → DA burst shifts to CS onset, not reward time.
3. **After learning, reward omission**: CS fires → $V(\text{CS}) > 0$; reward absent → $r_t = 0$ → $\delta_t = 0 + \gamma \cdot 0 - V(\text{CS}) < 0$ → DA dip at expected reward time.

All five models below preserve this signature while generalizing what is predicted and how.

---

## 1. Classic RPE Model (Sutton 1988)

### Goals and Motivation

The goal is to learn the **value function** $V(s_t) = \mathbb{E}[G_t \mid s_t]$: the expected discounted sum of future rewards starting from state $s_t$. Learning this purely from experience (not a model of the world) is the key challenge.

The key insight in Sutton (1988) is **bootstrapping**: instead of waiting until the end of an episode to compute $G_t$ exactly (Monte Carlo), use the current estimate of $V(s_{t+1})$ as a proxy target. This gives an online, incremental learning algorithm.

### Value Function and the Bellman Equation

From the definition of $G_t = r_t + \gamma G_{t+1}$, taking expectations:
$$V(s) = \mathbb{E}[r_t + \gamma V(s_{t+1}) \mid s_t = s]$$

This is the **Bellman consistency equation**. It says: the value of a state equals the expected immediate reward plus the discounted value of the next state. Any $V$ satisfying this for all states is the true value function.

### TD(0) Error and Learning

Define the **one-step TD error** as the difference between the Bellman target $r_t + \gamma V(s_{t+1})$ and the current prediction $V(s_t)$:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**Interpretation**: $\delta_t > 0$ means the outcome was better than predicted (positive surprise); $\delta_t < 0$ means worse than predicted (negative surprise); $\delta_t = 0$ means perfectly predicted.

The **tabular update rule**:
$$V(s_t) \leftarrow V(s_t) + \alpha \, \delta_t$$

This is a stochastic gradient step on the loss $\mathcal{L} = \frac{1}{2}(r_t + \gamma V(s_{t+1}) - V(s_t))^2$, treating $V(s_{t+1})$ as a fixed target (the "semi-gradient" trick that makes TD stable).

### Linear Function Approximation

For large or continuous state spaces, tabular representations are impractical. Instead, parameterize the value function linearly in features:
$$V(s_t) \approx \vec{w} \cdot \vec{\phi}_t = \sum_j w_j \phi_j(s_t)$$

where $\vec{\phi}_t = \vec{\phi}(s_t)$ is a fixed feature vector (e.g., tile coding, RBFs, one-hot encoding).

The TD error becomes:
$$\delta_t = r_t + \gamma \vec{w} \cdot \vec{\phi}_{t+1} - \vec{w} \cdot \vec{\phi}_t$$

The **semi-gradient weight update**:
$$\vec{w} \leftarrow \vec{w} + \alpha \, \delta_t \, \vec{\phi}_t$$

**Why $\vec{\phi}_t$ (not $\vec{\phi}_{t+1}$)?** The gradient of $V(s_t) = \vec{w} \cdot \vec{\phi}_t$ with respect to $\vec{w}$ is $\vec{\phi}_t$. The target's dependence on $\vec{w}$ through $\vec{\phi}_{t+1}$ is ignored (the "semi-gradient"), which is necessary for stability.

### Eligibility Traces: TD(λ)

The TD(0) error credits only the immediately preceding state. But rewards can follow states by many steps. **Eligibility traces** $\vec{e}_t$ accumulate recently visited features with exponential decay:
$$\vec{e}_0 = \vec{0}$$
$$\vec{e}_t = \gamma \lambda \, \vec{e}_{t-1} + \vec{\phi}_t$$

The weight update uses the trace instead of the instantaneous feature:
$$\vec{w} \leftarrow \vec{w} + \alpha \, \delta_t \, \vec{e}_t$$

where $\lambda \in [0,1]$ is the trace-decay parameter. $\lambda = 0$ recovers TD(0); $\lambda = 1$ approaches Monte Carlo. TD($\lambda$) generalizes across this spectrum by "soft-blending" n-step returns.

### Convergence and Properties

- Converges to $V^*$ under linear function approximation and a fixed policy if $\alpha$ decays appropriately (Tsitsiklis & Van Roy, 1997).
- The TD error signal has zero mean at convergence: $\mathbb{E}[\delta_t] = 0$ when $V = V^*$.
- The key Bellman contraction property: $\|T V - T V'\|_\infty \leq \gamma \|V - V'\|_\infty$ guarantees uniqueness.

### Neuroscience Mapping

| Biological Structure | Computational Role |
|----------------------|-------------------|
| Cortex | State features $\vec{\phi}_t$ |
| Striatum (MSNs) | Linear value readout $V(s_t) = \vec{w} \cdot \vec{\phi}_t$ |
| VTA/SNc DA neurons | TD error $\delta_t$ |
| Corticostriatal synapses | Weights $\vec{w}$ |
| DA → synaptic modulation | Weight update $\Delta w = \alpha \delta_t \phi_t$ |

---

## 2. Outcome-Specific Model: Distributional RL (Dabney et al. 2020)

### Goals and Motivation

The classic RPE model predicts the **mean** of future rewards, $\mathbb{E}[G_t]$. But the distribution of $G_t$ carries richer information (risk, variance, skewness). Distributional RL (Bellemare, Dabney, Munos 2017; Dabney et al. 2020) learns the **full return distribution** $Z(s)$, not just its mean.

The neuroscientific motivation: if different DA neurons have systematically different "reversal points" (the reward magnitude producing zero PE), this can be explained by distributional RL where each neuron $i$ estimates a different quantile $\tau_i$ of the return distribution.

### The Distributional Bellman Equation

Define $Z(s)$ as a random variable representing the return from state $s$. The distributional Bellman equation (equality in distribution):
$$Z(s) \stackrel{d}{=} R + \gamma Z(S')$$

where $S'$ is the next state and $\stackrel{d}{=}$ means "equal in distribution." This is more general than the scalar Bellman equation, which only asserts equality of expectations.

### Quantile and Expectile Regression

There are several ways to parameterize and learn return distributions. The version closest to neuroscience data (Dabney et al. 2020) uses **expectile regression**.

Each DA unit $i$ maintains a value estimate $V_i(s)$ corresponding to the $\tau_i$-expectile of $Z(s)$. The **expectile** generalizes the mean: if $\tau = 0.5$, the expectile equals the mean; larger $\tau$ pushes toward the upper tail (optimistic); smaller $\tau$ toward the lower tail (pessimistic).

The asymmetric loss function for expectile regression at level $\tau_i$:
$$\mathcal{L}_{\tau_i}(u) = |\tau_i - \mathbf{1}(u < 0)| \cdot u^2$$

This weights positive deviations by $\tau_i$ and negative deviations by $(1 - \tau_i)$.

### Asymmetric TD Update

For each unit $i$, define the TD error as usual:
$$\delta_i = r_t + \gamma V_i(s_{t+1}) - V_i(s_t)$$

But instead of a single learning rate, use **asymmetric rates** determined by $\tau_i$:
$$\alpha_i^+ = \alpha \cdot \tau_i \qquad \text{(for positive errors)}$$
$$\alpha_i^- = \alpha \cdot (1 - \tau_i) \qquad \text{(for negative errors)}$$

Update:
$$V_i(s_t) \leftarrow V_i(s_t) + \alpha_i^{\text{sgn}(\delta_i)} \, \delta_i$$

The effective expectile level is:
$$\tau_i = \frac{\alpha_i^+}{\alpha_i^+ + \alpha_i^-}$$

### Reversal Points

For each unit $i$, the **reversal point** is the reward magnitude $r^*_i$ such that the TD error is zero at steady state. Units with high $\tau_i$ (optimistic) will have high reversal points: they only produce negative PEs to large rewards. Units with low $\tau_i$ (pessimistic) have low reversal points: they respond negatively even to moderate rewards.

Formally, at the reversal point $r^*_i$:
$$r^*_i + \gamma V_i(s_{t+1}) - V_i(s_t) = 0$$

The distribution of reversal points across a population of DA neurons directly reflects the distribution of $\tau_i$ values, and thus the shape of the return distribution.

### Relationship to Classic RPE

The mean value across all channels converges to the classic RPE value:
$$\frac{1}{N} \sum_i V_i(s) \xrightarrow{} \mathbb{E}[Z(s)] = V(s)$$

and therefore:
$$\frac{1}{N} \sum_i \delta_i \xrightarrow{} \delta_{\text{total}}$$

The distributional model is an **outcome-specific** model in the Lee et al. taxonomy: each unit $i$ computes the same RPE structure (same state features, same reward $r_t$ as target), but with different weighting of the error sign.

### Key Neuroscience Prediction (Dabney 2020)

The model predicts that DA neurons should show:
- A distribution of reversal points across neurons (confirmed empirically)
- Pessimistic neurons (low $\tau_i$) with negative-skewed responses to moderate rewards
- Optimistic neurons (high $\tau_i$) with positive-skewed responses
- All neurons respond to unexpected reward (shared $r_t$ term) but with different magnitudes

**Critical distinction from feature-specific**: Distributional RL predicts uniform responses during cue periods (because all channels use the same features) but different responses at outcome time. Feature-specific predicts the opposite: heterogeneous cue responses, relatively uniform outcome responses.

---

## 3. Feature-Specific RPE Model (Lee et al. 2024)

### Goals and Motivation

Lee et al. 2024 address a specific empirical puzzle: VTA DA neurons show **heterogeneous** responses during task cue periods but relatively **homogeneous** responses to reward outcomes. Outcome-specific models (distributional RL, APE, SR) can explain heterogeneity at outcome time but predict uniform cue responses because all channels use the same state features. The feature-specific model inverts this logic.

The anatomical motivation: corticostriatal projections are highly **topographic** — different striatal medium spiny neurons (MSNs) receive preferential input from specific cortical regions. If different MSNs receive different cortical features, and different DA neurons are driven by different MSNs, then individual DA neurons will naturally compute prediction errors for different aspects of the current state.

### From Scalar to Distributed Value Representation

**Classic scalar model**: all cortical features $\vec{\phi}_t$ converge uniformly onto a single striatal value unit:
$$V(s_t) = \vec{w} \cdot \vec{\phi}_t = \sum_j w_j \phi_j$$

This requires complete, uniform convergence — anatomically unrealistic.

**Feature-specific model**: striatal unit $i$ receives preferential input from feature $\phi_{i,t}$ (a scalar component of the full feature vector). Its value prediction is:
$$V_{i,t} = w_i \, \phi_{i,t}$$

The **total value** is reconstructed by aggregating across channels:
$$V_{\text{total},t} = \sum_i V_{i,t} = \sum_i w_i \phi_{i,t} = \vec{w} \cdot \vec{\phi}_t$$

This is **algebraically identical** to the classic scalar value. The total circuit computes the same thing; only the intermediate representations differ.

### The Feature-Specific PE (Equation 1, Lee et al. 2024)

Each DA unit $i$, driven by striatal unit $i$, computes a **feature-specific prediction error**:
$$\boxed{\delta_{i,t} = \frac{r_t}{N} + \gamma V_{i,t+1} - V_{i,t} = \frac{r_t}{N} + w_i(\gamma \phi_{i,t+1} - \phi_{i,t})}$$

where $N$ is the number of channels.

### Derivation: Why $r_t / N$?

For the model to remain consistent with the classic scalar RPE, we require:
$$\sum_i \delta_{i,t} = \delta_{\text{total}} = r_t + \gamma V_{\text{total},t+1} - V_{\text{total},t}$$

Expanding the sum over the proposed $\delta_{i,t}$:
$$\sum_i \delta_{i,t} = \sum_i \left[\frac{r_t}{N} + w_i(\gamma\phi_{i,t+1} - \phi_{i,t})\right]$$
$$= N \cdot \frac{r_t}{N} + \sum_i w_i(\gamma\phi_{i,t+1} - \phi_{i,t})$$
$$= r_t + \gamma \sum_i w_i \phi_{i,t+1} - \sum_i w_i \phi_{i,t}$$
$$= r_t + \gamma V_{\text{total},t+1} - V_{\text{total},t} = \delta_{\text{total}} \checkmark$$

The factor $1/N$ ensures that the reward $r_t$ is split equally among channels so that their sum recovers the full reward signal. This is the **key conservation identity** of the model.

### Weight Update Rule

The corticostriatal weight for unit $i$ is updated by the local DA signal $\delta_{i,t}$ modulating the local cortical input $\phi_{i,t}$:
$$w_i \leftarrow w_i + \alpha \, \delta_{i,t} \, \phi_{i,t}$$

Note: In Lee et al.'s actual implementation (deep RL agent), weights are trained using the **summed** (scalar) RPE $\delta_{\text{total}}$, not the individual $\delta_{i,t}$. This is because they train with A2C, which uses the scalar critic. The feature-specific PEs are computed post-hoc for neural analysis. For a standalone implementation, using $\delta_{i,t}$ for local updates is the biologically meaningful choice.

### Predicted Response Signatures

**During cue periods** ($r_t = 0$):
$$\delta_{i,t} = w_i(\gamma\phi_{i,t+1} - \phi_{i,t})$$

Each DA unit reports the **time-derivative of its own feature**, weighted by how much that feature is associated with value ($w_i$). Different features change differently at cue onset → **heterogeneous cue responses**.

**At outcome time** ($r_t \neq 0$):
$$\delta_{i,t} = \frac{r_t}{N} + w_i(\gamma\phi_{i,t+1} - \phi_{i,t})$$

All units share the $r_t / N$ term → all respond to reward → **relatively uniform outcome responses** (with modulation from the feature term, unlikely to completely cancel $r_t/N$ for most units).

This asymmetry — heterogeneous during cue, uniform at outcome — is the **key empirical signature** that distinguishes feature-specific from outcome-specific models, and was confirmed in Engelhard et al. (2019) data.

### Feature-Based APE Extension (Eqs. 7–8, Lee et al. 2024)

The same logic applies if actions replace rewards:
$$\delta^a_{i,t} = \frac{I_{a_t=a}}{N} + \gamma V^a_{i,t+1} - V^a_{i,t} = \frac{I_{a_t=a}}{N} + w^a_{i,t}(\gamma\phi^a_{i,t+1} - \phi^a_{i,t})$$
$$w^a_t \leftarrow w^a_t + \eta \, \delta^a_t \, \phi^a_t$$

where $I_{a_t=a}$ is an indicator for whether the agent chose action $a$. This models DA neurons that encode action prediction errors within a distributed feature code.

---

## 4. Action Prediction Error Model (Greenstreet et al. 2025)

### Goals and Motivation

Greenstreet et al. (2025) report a striking finding: a subset of DA neurons in the dorsomedial striatum pathway encode prediction errors about **upcoming actions** (behavioral choices), completely independently of the reward value of those actions. This constitutes a "value-free" teaching signal that can shape action policies without reference to reward.

The conceptual key: in standard RL, rewards are external outcomes that the agent cannot control. Actions, by contrast, are generated by the agent's own policy. If the agent can predict its own upcoming behavior, deviations from that prediction form an action PE. This signal could serve as a "behavioral credit assignment" mechanism — it tells the brain when actions are more or less consistent with current behavioral tendencies.

### Formal Setup: Action as "Reward"

Treat the execution of action $a$ as if it were a "reward" signal. Define:
$$I_{a_t=a} = \begin{cases} 1 & \text{if action } a \text{ was taken at time } t \\ 0 & \text{otherwise} \end{cases}$$

The "value" of state $s$ for action $a$ is the expected future frequency of executing $a$:
$$V^a(s) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k I_{a_{t+k}=a} \;\middle|\; s_t = s\right]$$

This is not a reward value — it is the cumulative discounted number of times action $a$ is expected to be taken starting from state $s$.

### Scalar APE (Equations 5–6, Lee et al. 2024 formulation)

The **action prediction error** for a DA neuron with preferred action $a$:
$$\delta^a_t = I_{a_t=a} + \gamma V^a(s_{t+1}) - V^a(s_t) \tag{5}$$

The value update:
$$V^a(s_t) \leftarrow V^a(s_t) + \eta \, \delta^a_t \tag{6}$$

This is isomorphic to the classic scalar RPE with $I_{a_t=a}$ substituted for $r_t$.

**Predicted behavior**:
- When action $a$ is taken unexpectedly (high $V^a$ was not needed but action occurred): $\delta^a_t = 1 + \gamma V^a(s_{t+1}) - V^a(s_t) > 0$ → DA burst
- When action $a$ was strongly expected but a different action was taken: $\delta^a_t = 0 + \gamma V^a(s_{t+1}) - V^a(s_t) < 0$ → DA dip
- When action $a$ is taken as expected: $\delta^a_t \approx 0$ → no response

### Full TD Formulation vs. Rescorla-Wagner

Earlier APE implementations (Bogacz 2020; some Greenstreet preprints) used a simpler Rescorla-Wagner-style update (no future-predictive term):
$$V^a \leftarrow V^a + \eta(I_{a_t=a} - V^a)$$

Lee et al. (2024) extend this to the full TD formulation with the $\gamma V^a(s_{t+1})$ bootstrap term, making it consistent with the temporal-discounting framework. Rescorla-Wagner is a special case with $\gamma = 0$ (maximal temporal discounting — only the immediate action is predicted, not future action probabilities).

### Key Distinctions from Classic RPE

| | Classic RPE | Action PE |
|---|---|---|
| Prediction target | Future rewards | Future action frequency |
| "Reward" signal | External $r_t$ | Endogenous $I_{a_t=a}$ |
| Value-free? | No | **Yes** |
| Responds to reward alone? | Yes | No |
| Responds to action alone? | No | **Yes** |
| Drives reward learning? | Yes | Unclear/No |
| Drives policy learning? | Indirect | **Direct** |

### Greenstreet et al. 2025: Empirical Contributions

The 2025 paper (final published version of the 2022/2024 preprint) demonstrated:
- A population of DA neurons in DMS-projecting pathways signals action PEs while being insensitive to reward value.
- Optogenetic silencing of these neurons disrupts action-consistent behavior without affecting reward learning.
- The signal is characterized by: (a) response to action onset regardless of reward, (b) suppression when expected actions are omitted, (c) no modulation by reward magnitude.

---

## 5. SR Model: Dopamine as Generalized PE (Gardner et al. 2018)

### Goals and Motivation

Gardner, Schoenbaum & Gershman (2018) propose that phasic DA encodes a **generalized prediction error** that includes not just reward PEs but also PEs about state transitions. The framework is the **successor representation (SR)**, which separates the predictive structure of the environment (encoded in a "transition matrix") from the reward function.

The key insight: the classic RPE ($\delta = r + \gamma V(s') - V(s)$) can be **algebraically decomposed** as a reward-weighted sum of SR prediction errors. Different DA neurons could encode SR errors for different future states, which are then implicitly aggregated back into the classic RPE through reward weighting.

### Successor Representation Definition

For a fixed policy $\pi$, the **successor representation** $M(s, s')$ is the expected discounted future occupancy of state $s'$ when starting in state $s$:
$$M(s, s') = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k \mathbf{1}(s_{t+k}=s') \;\middle|\; s_t=s, \pi\right]$$

$M$ is a matrix: row $s$ gives the expected future visitation of all other states from state $s$. Diagonal entries are largest (you're already here), and off-diagonal entries decay with distance/probability.

### Value from SR

If $r(s')$ is the reward received at state $s'$, then:
$$V(s) = \sum_{s'} M(s, s') \, r(s') = M(s)^T \vec{r}$$

This **separates** the task structure (encoded in $M$) from the reward function (encoded in $\vec{r}$). This separation has a powerful computational advantage: if rewards change but transition structure stays the same, only $\vec{r}$ needs to be re-learned (rapid adaptation); if structure changes, $M$ must be updated.

### SR Bellman Equation and TD Errors

The SR satisfies its own Bellman equation:
$$M(s, s') = \mathbf{1}(s = s') + \gamma \sum_{s''} P(s'' \mid s) M(s'', s')$$

For each target state $s'$, define the **SR prediction error**:
$$\delta^M_t(s') = \mathbf{1}(s_t = s') + \gamma M(s_{t+1}, s') - M(s_t, s')$$

This is a **vector** of prediction errors — one per state $s'$. The full SR PE is:
$$\vec{\delta}^M_t = \vec{e}_{s_t} + \gamma M(s_{t+1}, :) - M(s_t, :)$$

where $\vec{e}_{s_t}$ is a one-hot vector at the current state.

The SR update rule:
$$M(s_t, :) \leftarrow M(s_t, :) + \alpha \, \vec{\delta}^M_t$$

### Algebraic Equivalence to Classic RPE

The **reward-weighted** SR PE recovers the classic RPE:
$$\sum_{s'} r(s') \, \delta^M_t(s') = r(s_t) + \gamma V(s_{t+1}) - V(s_t) = \delta_{\text{classical}}$$

**Proof**:
$$\sum_{s'} r(s') \left[\mathbf{1}(s_t=s') + \gamma M(s_{t+1}, s') - M(s_t, s')\right]$$
$$= r(s_t) + \gamma \sum_{s'} r(s') M(s_{t+1}, s') - \sum_{s'} r(s') M(s_t, s')$$
$$= r(s_t) + \gamma V(s_{t+1}) - V(s_t) = \delta_{\text{classical}} \checkmark$$

### Successor Features (Generalization)

For continuous or high-dimensional state spaces, replace states with feature vectors $\vec{\phi}(s)$ to get **successor features**:
$$\vec{\psi}(s) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k \vec{\phi}(s_{t+k}) \;\middle|\; s_t=s\right]$$

Value from successor features:
$$V(s) = \vec{w}^T \vec{\psi}(s)$$

where $\vec{w}$ maps features to reward (learned by reward regression).

Successor feature PE (vector):
$$\vec{\delta}^{\psi}_t = \vec{\phi}(s_{t+1}) + \gamma\vec{\psi}(s_{t+1}) - \vec{\psi}(s_t)$$

In the Lee et al. (2024) SR model implementation (Eqs. 2–4):
$$SF_{i,t} = \vec{\phi}_t \cdot \vec{w}_i \tag{2}$$
$$\delta_{i,t} = \phi_{i,t} + \gamma SF_{i,t+1} - SF_{i,t} \tag{3}$$
$$\vec{w}_i \leftarrow \vec{w}_i + \alpha \, \delta_{i,t} \, \vec{\phi}_t \tag{4}$$

Here each channel $i$ learns to predict the future trajectory of its own feature $\phi_i$. The resulting $\delta_{i,t}$ is a "sensory PE" about unexpected changes in feature $i$.

### SR vs. Feature-Specific: Key Difference

| Dimension | SR Model | Feature-Specific RPE |
|---|---|---|
| What is predicted? | **Future** feature values $\vec{\phi}(s_{t+k})$ | **Current** feature values $\vec{\phi}_t$ |
| PE at cue time | $\phi_{i,t} + \gamma SF_{i,t+1} - SF_{i,t}$ | $w_i(\gamma\phi_{i,t+1} - \phi_i)$ |
| Responds to confirmatory cues? | Weakly (disconfirmatory > confirmatory) | Yes (confirmatory > disconfirmatory) |
| At outcome time | Inconsistently modulated by reward | All units respond to $r_t/N$ |
| Taxonomy | Outcome-specific | Feature-specific |

Lee et al. (2024) show that the SR model, while producing heterogeneous cue responses, fails to replicate the confirmatory cue preference and reward-modulated outcome responses seen in VTA DA data.

---

## 6. Custom Weighted-β Model

### Design Rationale

Your model is an extension of Lee et al.'s feature-specific RPE that introduces **heterogeneous contribution weights** $\beta_i$ for each channel. The motivation is to allow different striatal units to contribute unequally to the total value signal, capturing the biological reality that:
- Some corticostriatal pathways may have stronger synaptic weight into the DA computation
- Different features may have different "relevances" to the global value signal
- The model can be generalized to the baiting task where different sides have different baseline reward probabilities (analogous to different $\beta_i$ values)

### Full Mathematical Formulation

**Per-unit value** (unit $i$ receives scalar feature $\phi_{i,t}$, has scalar weight $w_i$):
$$V_{i,t} = w_i \cdot \phi_{i,t}$$

**Total value** (weighted sum):
$$V_{\text{total},t} = \sum_i \beta_i V_{i,t} = \sum_i \beta_i w_i \phi_{i,t}$$

**Total TD error** (standard RPE over total value):
$$\delta_{\text{total}} = r_t + \gamma V_{\text{total},t+1} - V_{\text{total},t}$$

**Unit-specific PE**:
$$\boxed{\delta_{i,t} = \frac{r_t}{\beta_i N} + w_i(\gamma\phi_{i,t+1} - \phi_{i,t})}$$

### Derivation: Consistency Identity

We want $\sum_i \beta_i \delta_{i,t} = \delta_{\text{total}}$. Verify:
$$\sum_i \beta_i \delta_{i,t} = \sum_i \beta_i \left[\frac{r_t}{\beta_i N} + w_i(\gamma\phi_{i,t+1} - \phi_{i,t})\right]$$
$$= \sum_i \frac{r_t}{N} + \sum_i \beta_i w_i(\gamma\phi_{i,t+1} - \phi_{i,t})$$
$$= r_t + \gamma\sum_i \beta_i w_i \phi_{i,t+1} - \sum_i \beta_i w_i \phi_{i,t}$$
$$= r_t + \gamma V_{\text{total},t+1} - V_{\text{total},t} = \delta_{\text{total}} \checkmark$$

The $\beta_i$ cancel in the reward term (because $\beta_i \cdot \frac{r_t}{\beta_i N} = \frac{r_t}{N}$, independent of $\beta_i$), meaning the reward is always split equally across channels regardless of $\beta_i$. The $\beta_i$ weighting only enters through the value-prediction term.

### Interpretation of $\beta_i$

**Effect on reward signal received**: $r_t / (\beta_i N)$
- Units with **large** $\beta_i$ receive a **smaller** reward signal per unit
- Units with **small** $\beta_i$ receive a **larger** reward signal per unit
- Intuitively: high-$\beta_i$ units contribute more to total value through their feature predictions; they "need" less direct reward to stay calibrated

**Effect on value contribution**: $\beta_i V_{i,t}$
- High-$\beta_i$ units have amplified contributions to total value
- This makes them more "powerful" value-shapers — their weight $w_i$ has a larger effect on behavior through value-guided action selection

**Biological analogy**: $\beta_i$ could represent the density or efficacy of the striatal-to-DA projection for channel $i$ — how much DA activity is driven by unit $i$'s value readout.

### Update Rules

**Option 1: Local update** (biologically most plausible — each unit updates by its own local PE):
$$w_i \leftarrow w_i + \alpha \, \delta_{i,t} \, \phi_{i,t}$$

**Option 2: Global update** (uses the full scalar RPE — biologically plausible if DA broadcasts diffusely):
$$w_i \leftarrow w_i + \alpha \, \delta_{\text{total}} \, \phi_{i,t}$$

**Option 3: Scaled local update** (gradient of $\mathcal{L} = \frac{1}{2}\delta_{\text{total}}^2$ w.r.t. $w_i$):

The proper gradient is:
$$\frac{\partial \mathcal{L}}{\partial w_i} = -\delta_{\text{total}} \cdot \frac{\partial V_{\text{total},t}}{\partial w_i} = -\delta_{\text{total}} \cdot \beta_i \phi_{i,t}$$

Gradient descent:
$$w_i \leftarrow w_i + \alpha \, \delta_{\text{total}} \cdot \beta_i \, \phi_{i,t}$$

But since $\delta_{\text{total}} = \sum_j \beta_j \delta_{j,t}$, this becomes:
$$w_i \leftarrow w_i + \alpha \left(\sum_j \beta_j \delta_{j,t}\right) \beta_i \, \phi_{i,t}$$

The choice of update rule has important implications for learning dynamics and biological plausibility. **Option 1 is recommended** for biological modeling; **Option 3 is recommended** for normative/optimization analysis.

### Comparison with Lee et al. Feature-Specific Model

| Aspect | Lee et al. (2024) | Custom β Model |
|---|---|---|
| Value | $V_{\text{total}} = \sum_i V_{i,t}$ | $V_{\text{total}} = \sum_i \beta_i V_{i,t}$ |
| Unit PE | $\delta_{i,t} = \frac{r_t}{N} + w_i(\gamma\phi_{i,t+1} - \phi_{i,t})$ | $\delta_{i,t} = \frac{r_t}{\beta_i N} + w_i(\gamma\phi_{i,t+1} - \phi_{i,t})$ |
| Reward attribution | Equal across all units | Inversely weighted by $\beta_i$ |
| Feature contribution to value | Equal $\beta_i = 1$ for all $i$ | Heterogeneous $\beta_i$ |
| Extra parameter | None | $\{\beta_i\}_{i=1}^N$ |
| Reduces to Lee et al. | — | When all $\beta_i = 1$ |

### Optional: Learning $\beta_i$

The $\beta_i$ weights could themselves be learned (meta-learning). The gradient of $\mathcal{L}$ with respect to $\beta_i$:
$$\frac{\partial \mathcal{L}}{\partial \beta_i} = -\delta_{\text{total}} \cdot V_{i,t}$$

Update:
$$\beta_i \leftarrow \beta_i + \alpha_\beta \, \delta_{\text{total}} \cdot V_{i,t}$$

To prevent degeneracy (all $\beta_i \to \infty$ or $\to 0$), enforce the constraint $\sum_i \beta_i = N$ (equivalently $\frac{1}{N}\sum_i \beta_i = 1$) using a projected gradient or softmax reparameterization:
$$\beta_i = N \cdot \frac{\exp(b_i)}{\sum_j \exp(b_j)}$$

and learn $b_i$ instead. This keeps $\sum_i \beta_i = N$ always.

---

## 7. Cross-Model Summary Table

| | Classic RPE | Distributional | Feature-Specific | Action PE | SR | Custom β |
|---|---|---|---|---|---|---|
| What is predicted? | $\mathbb{E}[G_t]$ | Distribution of $G_t$ | Same as classic | Future action frequency | Future state occupancy | $\mathbb{E}[G_t]$ (weighted) |
| Source of heterogeneity | None | Optimism level $\tau_i$ | Feature assignment $\phi_{i,t}$ | Preferred action $a$ | Future state $s'$ | Feature + weight $\beta_i$ |
| Taxonomy | Baseline | Outcome-specific | Feature-specific | Outcome-specific | Outcome-specific | Feature-specific (extended) |
| Cue responses | Uniform | Mostly uniform | **Heterogeneous** | Heterogeneous | Heterogeneous | **Heterogeneous** |
| Outcome responses | Uniform | **Heterogeneous** | Mostly uniform | Not reward-driven | Variable | Mostly uniform |
| Value-free? | No | No | No | **Yes** | Partial | No |
| $\sum_i \delta_i$ = classic? | trivially | Approximately | Exactly | By substitution | Weighted | Exactly ($\sum_i \beta_i \delta_i$) |

---

## 8. Grid-World Task Design

### Environment Specification

**Motivation for grid world**: A grid-world navigation task is the simplest possible non-trivial RL environment with the relevant structure for testing all five models. It has:
- A well-defined state space (grid positions)
- Sparse but clear rewards
- Block structure (goal location changes)
- Interpretable features with graded spatial tuning

**Grid structure**:
- $H \times W$ grid (e.g., $7 \times 7$)
- States: $s = (r, c)$ with $r \in \{0,...,H-1\}$, $c \in \{0,...,W-1\}$
- Actions: $\mathcal{A} = \{\text{North, South, East, West}\}$
- Walls at boundaries (agent cannot leave grid)
- Start: fixed position (e.g., center) or random within a region

**Reward structure**:
- Single goal location $G_b$ per block $b$
- Reward $r = +1$ upon reaching $G_b$, episode ends
- Zero reward at all other positions
- Each episode ends either at goal (success) or after $T_{\max}$ steps (truncation)

**Block structure** (this is the key for testing adaptation):
- Block $b$ lasts $K$ episodes (e.g., $K = 30$)
- Goal switches between a fixed set of candidate locations $\{G_1, G_2, G_3, G_4\}$ (e.g., four corners or four fixed positions)
- Schedule: $G_1 \to G_2 \to G_3 \to G_2 \to G_1 \to ...$ (reversals) or random

**Action selection**: Softmax policy over Q-values or V + advantage, with temperature parameter $\beta_{\text{policy}}$ controlling exploration.

### Feature Representations

The choice of features $\phi_{i,t}$ is the critical modeling decision. Each model needs a feature set suited to its structure.

#### For Classic RPE (tabular features)

One-hot encoding of current state position:
$$\phi_j(s) = \mathbf{1}(s = s_j), \quad j \in \{1,...,H \times W\}$$

This gives a tabular representation. $V(s) = \vec{w} \cdot \vec{\phi}(s) = w_j$ for $s = s_j$. Equivalent to a lookup table.

#### For Feature-Specific Model (spatially tuned features)

**Option A — Gaussian place fields** (most biologically plausible):
$$\phi_i(s) = \exp\left(-\frac{\|s - c_i\|^2}{2\sigma^2}\right)$$

where $c_i$ are centers tiling the grid and $\sigma$ controls tuning width. Channel $i$ has maximum response when the agent is at center $c_i$.

**Option B — Direction/distance features** (most interpretable for block switching):
For each candidate goal location $G_k$ ($k = 1,...,K_G$), define four features encoding direction to $G_k$:
$$\phi_{k,\text{north}}(s) = \max(0, G_k^r - s^r) / d_{\max}$$
$$\phi_{k,\text{south}}(s) = \max(0, s^r - G_k^r) / d_{\max}$$
$$\phi_{k,\text{east}}(s) = \max(0, G_k^c - s^c) / d_{\max}$$
$$\phi_{k,\text{west}}(s) = \max(0, s^c - G_k^c) / d_{\max}$$

This creates a feature for "how far north is goal $k$ from here," etc. These features are directly interpretable: when the goal switches from $G_1$ to $G_2$, the distance-to-$G_1$ features become irrelevant and distance-to-$G_2$ features become predictive of reward.

**Option C — Successor feature-like** (for SR model):
Features derived from position coding, but the SR model additionally learns to predict future feature values. For the grid world, the SR $M(s, s')$ captures which states are likely to be visited in the future — which encodes the geometry of the maze independently of reward.

#### For Action PE Model

Features encode the agent's current state and action tendency:
$$\phi^a_i(s) = \mathbf{1}(\text{preferred direction from state } s = a_i)$$

Or spatial features but the "reward" signal is replaced by $I_{a_t = a}$.

### Connecting to the Baiting Task

The grid-world task is a stepping stone to the two-armed baiting task. The structural correspondence is:

| Grid-World | Baiting Task |
|---|---|
| Goal location | Rewarded side (left/right) |
| Block (goal changes) | Block (reward probabilities change) |
| Distance to candidate goals $G_k$ | Prior probability that side $k$ pays off |
| $\beta_i$ in custom model | Relative baseline probability of side $i$ |
| Navigation features $\phi_{i,t}$ | Internal belief state about each side |
| Within-block learning | Within-block choice updating |
| Block switch | Contingency reversal |

In the baiting task, the $\beta_i$ weights in the custom model have a natural interpretation: they reflect the agent's representation of the **baseline reward rate** on side $i$. A side with higher baseline probability should have a larger $\beta_i$, meaning its value prediction is weighted more heavily in driving choices, but also meaning it receives less direct reward signal (because it already "expects" reward often). This inverse relationship captures the normalization of prediction errors by prior expectations.

### Model-Specific Learning Dynamics in the Grid World

**Classic RPE**: Values spread from the goal location backwards along the optimal path. After a block switch, the old goal location has high $V$ but gives no reward → large negative PEs → gradual unlearning. Slow adaptation unless learning rate is high.

**Distributional RL**: Different quantile channels adapt at different rates. Optimistic channels ($\tau_i \approx 1$) maintain high $V_i$ even after a few unrewarded visits; pessimistic channels ($\tau_i \approx 0$) rapidly detect the block switch. The spread of reversal points provides a natural "memory" of the distribution of outcomes.

**Feature-Specific RPE**: If features include distance to each candidate goal, then features of the new goal immediately start changing when the agent navigates toward/away from it. The weights $w_i$ for new-goal features begin updating right away. This could enable **faster adaptation** to block switches than the classic RPE, because relevant features are already changing even before the agent reaches (or misses) the goal.

**Action PE**: Does not directly encode reward — encodes how well the current policy is being predicted. At a block switch, if the agent continues using the old policy (heading to old goal), the action PE signals are internally consistent (actions match policy) but the reward RPE signals are inconsistent. The interaction between APE and RPE signals may be key to how policy updating and value updating interact.

**SR Model**: The successor representation encodes the transition structure, which is **block-invariant** (maze structure doesn't change). The SR $M(s, s')$ only needs to be updated if the agent's policy changes. Only the reward weights $\vec{w}$ (mapping features to value) need updating at a block switch. This should give **faster block adaptation** than the classic RPE, which must re-learn the full value function.

**Custom β Model**: The $\beta_i$ can be set to reflect a prior over which locations are likely to be goals. If the possible goal locations are known, setting $\beta_i$ proportional to the prior probability of goal $G_i$ biases the total value toward the most likely goal locations. When the block switches, the mismatch between $\beta_i$-weighted predictions and actual outcomes will drive strong updates in the channels corresponding to the new goal's features.

---

## 9. Key Conceptual Questions for Model Comparison

When running the grid-world simulations, the following questions will differentiate the models:

**Q1: Adaptation speed after block switch**
How many trials after the goal switches does the agent need before performing well? SR should adapt fastest (structure unchanged); classic RPE slowest; feature-specific intermediate; custom β depends on $\beta_i$ configuration.

**Q2: Neural response profiles at cue vs. outcome**
If you track the $\delta_{i,t}$ signals across channels: feature-specific and custom β should show heterogeneous cue responses, uniform outcome responses. Distributional should show uniform cue, heterogeneous outcome.

**Q3: Transfer across blocks**
Does the agent reuse information from block 1 in block 3 (same goal location)? SR should show positive transfer through its stored $M$; classic RPE may show positive transfer if $\alpha$ is low enough; feature-specific depends on whether goal-specific features persist.

**Q4: Effect of $\beta_i$ in the custom model**
Compare $\beta_i = 1$ for all $i$ (= Lee et al.) to $\beta_i$ proportional to goal prior. The prior-informed $\beta$ should speed learning when the priors are correct but slow adaptation when they're wrong (prior-induced bias). This directly parallels the baiting task's block structure.

**Q5: Value function geometry**
Visualize $V(s)$ across the grid at different learning stages. For the feature-specific and custom β models, the geometry reflects the learned feature weights $w_i$, not just distance from the goal. What spatial structure do the $\delta_{i,t}$ signals show?

---

*This guide provides the mathematical foundations. Implementation details (code, simulation parameters, analysis pipelines) are addressed separately.*
