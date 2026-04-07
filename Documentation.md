# HRL Toolkit Documentation

This Markdown document provides a structured overview of the Python modules
available in the workspace.  It is intended to help developers navigate the
hierarchical reinforcement learning (HRL) codebase, understand key classes and
functions, and see how components interact.

## Purpose of the code

This toolkit implements **Hierarchical Reinforcement Learning (HRL)** using the **Option-Critic Architecture**.

The code combines two levels of learning:
1. **High-level policy**: Learns to select discrete **options** (sub-policies/skills) and when to switch between them
2. **Low-level policy**: Learns to execute continuous actions optimally under a given option

The key benefit is that high-level learning can operate on a longer timescale (deciding options less frequently)
while low-level learning handles rapid decision-making (continuous action control), enabling more efficient learning
of complex, hierarchical behaviors.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              OptionAgent (High-Level)               │
│  • Selects discrete options      [OptionValue Net]  │
│  • Manages option termination    [Termination Net]  │
│  • Coordinates training            (HRL Logic)      │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        V                     V
┌────────────────────────────────────────┐
│     Low-Level Agent (Continuous RL)    │
│  • TD3 or DDPG algorithm               │
│  • State is augmented with option      │
│  • Learns continuous actions           │
│  • Uses Actor & Critic networks        │
└────────────────┬───────────────────────┘
                 │
        ┌────────┴────────┐
        V                 V
┌──────────────────┐  ┌──────────────────┐
│   Actor Net      │  │   Critic Net     │
│ (Action Policy)  │  │ (Value Function) │
└──────────────────┘  └──────────────────┘
```

## Supported Low-Level Algorithms

- **DDPG** (Deep Deterministic Policy Gradient): Single actor-critic pair, simpler but can be less stable
- **TD3** (Twin Delayed DDPG): Two critics, delayed actor updates, and policy smoothing - more stable and sample-efficient

## Training Workflow

The HRL training pipeline follows these steps:

### Initialization Phase

1. **Environment Setup**: Initialize Gymnasium environment (e.g., MuJoCo, Custom)

2. **Algorithm Selection**: User specifies low-level algorithm:
   - Command line argument `--algo {ddpg, td3}`
   - Instantiate corresponding low-level agent (DDPG or TD3) with hyperparameters
   - Both algorithms share the same interface

3. **High-Level Agent**: Create OptionAgent instance (`agent`)
   - Passes the low-level agent as a component
   - Initializes OptionValue network (for option selection)
   - Initializes Termination network (for option switching)
   - Sets up optimizers and target networks

4. **Replay Buffer**: Initialize ReplayBuffer
   - Stores transitions: (state, action, reward, next_state, done, **option**, **terminated**)
   - The option and terminated flags enable HRL-specific learning

### Episode Loop (repeated for multiple episodes)

5. **Episode Reset**: `agent.reset(initial_observation)`
   - Selects initial option for the episode
   - An initial state is chosen based on the initial option 
   - Resets step counters

6. **Interaction Loop** (repeated for T timesteps per episode):

   a. **Option Handling & Action Selection**: `action, option, terminated, term_steps = agent.act(obs, noise_std, greedy_option)`
      - Checks if current option should terminate using β(s, o) from termination network
      - If termination: switches to new option via option value network
      - Low-level agent selects action: augmented_obs = [obs, one_hot(option)]
      - Returns action, current option, and termination flag
   
   b. **Environment Step**: Execute action in environment
      - Observe: next_state, reward, done (episode terminated)
      - Handle environment-specific reward shaping
      - Reward can be augmented with deliberation cost (penalty for switching options)
   
   c. **Replay Buffer**: Store transition
      - `buffer.push(obs, action, reward, next_obs, done, option=option, terminated=terminated)`
      - Note: `done` indicates episode termination; `terminated` indicates option termination
   
   d. **Training Update** (when buffer has enough samples):
      - `low_level_losses, optv_loss, term_loss = agent.update(buffer, batch_size=256, updates=per_step)`
      - Updates high-level networks (OptionValue and Termination)
      - Updates low-level networks (Actor and Critic)

### Key Data Structures

**Replay Buffer Transition:**
- `obs`: Current state
- `action`: Action executed
- `reward`: Reward (may include deliberation cost)
- `next_obs`: Resulting state
- `done`: Whether episode ended
- `option`: Discrete option index (K choices)
- `terminated`: Whether option terminated via β(s, o)

**Option Value Q(s, o):**
- Estimates expected return when executing option o in state s
- Used to select options greedily: o* = argmax_o Q(s, o)

**Termination Probability β(s, o):**
- Probability that option o terminates in state s ∈ [0, 1]
- If high: option likely to switch soon
- If low: option likely to continue
- Trained to maximize time in good options (high Q-value) and minimize time in bad ones

---



## utils.py

Utility helpers used across the project:

- **`set_seed(seed)`**
  - Sets the same random seed for Python's `random`, NumPy, PyTorch, and the
    Gym environment to ensure reproducible runs.

- **`make_env(env_name, seed=0)`**
  - Constructs a Gymnasium environment, resets it with the provided seed, and
    seeds its action space.

- **`to_tensor(obs, device=None)`**
  - Converts an observation to a `torch.FloatTensor` on the specified device.

- **`one_hot_option(option, num_options)`**
  - Returns a one-hot encoded NumPy array representing a discrete option index.

- **`soft_update(net, net_targ, tau)`**
  - Performs the soft update of parameters: \(\theta_{targ} \leftarrow
    (1-\tau)\theta_{targ}+\tau\theta\). Used for slowly-changing target
    networks.

- **`moving_average(x, window=50)`**
  - Computes a fixed-window moving average over a sequence of numbers.

- **`save_plots(run_dir, rows, ma_window=50)`**
  - Generates and saves several training plots (returns, losses, HRL stats,
    option durations) based on a list of per-episode dictionaries.

---

## networks.py

Neural network architectures used by both low-level and high-level agents.

All networks use layer normalization and ReLU activations for stability and better gradient flow.


### class Actor(nn.Module):

Actor network that maps an augmented observation (state + option) to a continuous action in the valid action space.

**Architecture:**
- Fully connected layers with layer normalization 
- Input: augmented observation (obs_dim + option dimension)
- Output: continuous action scaled to action limits using tanh

**Method: `forward(x)`**
- Takes a batch of observations `x` of shape (B, obs_dim)
- Returns actions of shape (B, act_dim) scaled to [-act_limit, act_limit]

**Key Parameters:**
- `obs_dim`: dimension of the observation space
- `act_dim`: dimension of the action space
- `act_limit`: upper/lower bound for valid actions (e.g., 2.0 for pendulum)
- `hidden`: size of hidden layers (default 256)


### class Critic(nn.Module):

Critic network that evaluates the Q-value Q(state, action) estimating how good an action is in a given state.

**Architecture:**
- Takes both observation and action as input (concatenated after first layer)
- Two fully connected layers with layer normalization and ReLU
- Single scalar output representing Q-value

**Method: `forward(x, a)`**
- Takes observation batch `x` of shape (B, obs_dim) and action batch `a` of shape (B, act_dim)
- Returns Q-values of shape (B, 1)

**Key Parameters:**
- `obs_dim`: dimension of the observation space
- `act_dim`: dimension of the action space
- `hidden`: size of hidden layers (default 256)


### class OptionValue(nn.Module):

High-level network that estimates Q-values for all available options given a state: Q(state, option).

Used by the OptionAgent to select which option to execute based on state.

**Architecture:**
- Maps observation to Q-values for each option
- Input: observation (obs_dim)
- Output: Q-values for all K options (num_options)

**Method: `forward(x)`**
- Takes observation batch `x` of shape (B, obs_dim)
- Returns Q-values of shape (B, num_options), one Q-value per option

**Key Parameters:**
- `obs_dim`: dimension of the observation space
- `num_options`: number of discrete options (K)
- `hidden`: size of hidden layers (default 256)


### class Termination(nn.Module):

Termination network that outputs the probability of terminating each option in a given state: β(state, option).

Used by OptionAgent to decide when to switch to a different option.

**Architecture:**
- Maps observation to termination logits for all K options
- Input: observation (obs_dim)
- Output: logits for termination probability of each option

**Method: `forward(s)`**
- Takes observation batch `s` of shape (B, obs_dim)
- Returns logits of shape (B, num_options) - one logit per option

**Method: `beta(s, option)`**
- Takes observation batch `s` and a batch of option indices `option`
- Returns sigmoid-transformed termination probabilities: β ∈ [0, 1]
- If β is close to 1: option likely to terminate
- If β is close to 0: option likely to continue

**Key Parameters:**
- `obs_dim`: dimension of the observation space
- `num_options`: number of discrete options (K)
- `hidden`: size of hidden layers (default 256)


## OptionAgent

This script incorporate the logic of HRL above the continuous RL in low level (DDPG/TD3)
It's purpose is to choice a discrete option using an "options policy" and deciding when to change the option using a specific termination function.

Everything is handled by the class `OptionAgent`.

Class Attributes: 

| Attribute | Meaning  | 
|----------|----------|
| obs_dim | state dimension | 
| num_options| number of discrete options K|
| low_level_agent | low level RL agent | 
| device | | 
| tau |  | 
| option_value | OptionaValue Network  |
| option_value_targ | OptionaValue Target Network | 
| option_value_opt| Optimizer: Adam by default | 
| self.mse | MSE loss| 
| termination  | Termination Network | 
| delib_cost  | Deliberation Cost | 
| eps_option | optional epsilon-greedy on option selection |
| terminate_deterministic | if True, terminate when beta>0.5 instead of sampling Bernoulli | 
| current_option |  | 
| num_terminations  | | 
| num_option_switches |  | 
| min_option_steps  | | 

### `def select_option(self, obs, greedy=False):`

Select a discrete option given the current observation using the OptionValue network.

**Behavior:**
- Uses the `option_value` network to compute Q(s, o) for all options
- If `greedy=True`: selects option with highest Q-value using `argmax`
- If `greedy=False`: uses epsilon-greedy strategy:
  - With probability `eps_option`: randomly select any option (exploration)
  - Otherwise: select option with highest Q-value (exploitation)

**Parameters:**
- `obs`: observation (numpy array or tensor)
- `greedy`: if True, use deterministic greedy selection; if False, use epsilon-greedy

**Returns:**
- Integer option index in range [0, num_options)


### `def should_terminate(self, obs, option):`

Determine whether the current option should terminate based on the termination network.

Uses the termination function β(s, o) ∈ [0, 1] which predicts the probability of terminating option o in state s.

**Behavior:**
- Calls `termination.beta(s, o)` to get termination probability β
- If `terminate_deterministic=True`: terminates if β > 0.5 (hard threshold)
- If `terminate_deterministic=False`: samples termination as Bernoulli(β) (stochastic)

**Parameters:**
- `obs`: observation (numpy array or tensor)
- `option`: current option index

**Returns:**
- Boolean: True if option should terminate, False otherwise


### `def reset(self, obs=None):`

Initialize the agent at the start of a new episode.

**Behavior:**
- Sets `option_steps` counter to 0
- Sets `current_option` to None
- If `obs` is provided: immediately selects a new option using `select_option(obs)`
  and increments `num_option_switches` (in theory obs will always be provided at episode start)

**Parameters:**
- `obs`: (optional) initial observation; if provided, an option is selected

**Note:** Call this function at the beginning of each episode before the first `act()` call.


### `def act(self, obs, noise_std=None, greedy_option=False):`

Select and execute an action, handling option termination and low-level action selection.

**Process:**
1. If no option is active (`current_option` is None): select a new option
2. Check if current option should terminate (only if `option_steps >= min_option_steps`):
   - If yes: select a new option, reset step counter, set `did_terminate=True`
3. Call low-level agent to select action: `action = low_level.act(augmented_obs, option, noise_std)`
4. Increment `option_steps`
5. Return action and option metadata


**Parameters:**
- `obs`: current observation
- `noise_std`: exploration noise scale for low-level agent
- `greedy_option`: if True, use greedy option selection; if False, use epsilon-greedy

**Returns (tuple):**
- `action`: numpy array of shape (act_dim,) - action to execute in environment
- `option`: int - the current/active option
- `did_terminate`: bool - whether the option terminated at this step
- `term_steps`: int or None - number of steps the terminated option lasted (for monitoring)

**Key Features:**
- `min_option_steps`: options must run for at least this many steps before terminating
- Updates counters: `num_terminations`, `num_option_switches`, `option_steps`


### `def update(self, replay_buffer, batch_size=256, update_iteration=1):`

Update the high-level (OptionValue and Termination) and low-level networks using samples from the replay buffer.

**High-Level Component - OptionValue Network Update:**

Trains Q(s, o) using TD learning with option-aware targets:

- If option terminates at s': use best next option → target = Q(s', o*)  where o* = argmax_o' Q(s', o')
- If option continues: use same option → target = Q(s', o)

The TD target incorporates both cases using the termination probability β(s', o):
$$\text{target} = r + \gamma (1 - \beta(s', o)) \cdot Q(s', o) + \gamma \beta(s', o) \cdot \max_{o'} Q(s', o') \cdot (1 - \text{done})$$

**High-Level Component - Termination Network Update:**

Learns when to switch options by maximizing the advantage-weighted termination:

$$\mathcal{L}_{\text{term}} = \mathbb{E}[\beta(s', o) \cdot (A(s', o) + c)]$$

where:
- $A(s', o) = Q(s', o) - V(s')$ is an advantage signal (motivation to switch)
- $V(s') = \max_{o'} Q(s', o')$ is the value of the best next option
- $c$ is `delib_cost` (penalty for switching) to prevent excessive option changes

**Low-Level Component:**

Calls `low_level_agent.update()` to train the continuous control policy (DDPG/TD3) as usual.

**Parameters:**
- `replay_buffer`: experience replay buffer with stored transitions
- `batch_size`: number of samples per update step
- `update_iteration`: number of update steps per call

**Returns (tuple):**
- `low_level_out`: output from low-level agent update (losses, etc.)
- `optv_loss_mean`: mean OptionValue loss across all update iterations
- `term_loss_mean`: mean Termination loss across all update iterations

**Training Details:**
- Soft-updates target network: `option_value_targ ← (1-τ)option_value_targ + τ option_value`
- Gradient clipping applied to termination network (max_norm=1.0)
- All losses monitored and returned for logging


### `def get_stats():`

Return current statistics about the agent's high-level behavior.

**Returns (dictionary):**
- `current_option`: currently active option index
- `option_steps`: steps elapsed since current option selection
- `num_terminations`: total number of option terminations so far
- `num_option_switches`: total number of option switches (including initial selection)



## Low-Level Agents 

Low-level agents implement continuous control algorithms (DDPG or TD3) that learn to execute actions within a given option. 
The key difference from standard DDPG/TD3 is that the state is **augmented** with the current option (concatenated as one-hot encoding), 
so the policy becomes: `action = π(s_aug) = π([state, one_hot(option)])`.

Both agents follow the same interface:
- `act(obs, noise_std, option)`: Select action given observation and current option
- `update(replay_buffer, batch_size, update_iteration)`: Update networks from replay buffer samples
- `augment_obs(obs, option)`: Concatenate one-hot option encoding to state


### class DDPG

Deep Deterministic Policy Gradient (DDPG) is a continuous control RL algorithm.

**Key Components:**
- **Actor**: Maps augmented state [state, one_hot(option)] to action
- **Critic**: Maps [state, action, one_hot(option)] to Q-value
- **Target Networks**: Slowly-updated copies of actor and critic for stability
- **Experience Replay**: Samples random batches from buffer to break temporal correlations

**Class Attributes:**

| Attribute | Meaning  | 
|----------|----------|
| obs_dim | state dimension from environment | 
| act_dim | action dimension (continuous)  |
| act_limit | upper/lower bound for actions |
| num_options | number of discrete options (K) |
| obs_dim_aug | augmented state dimension = obs_dim + num_options |
| actor | Actor network |
| actor_targ | Target actor network (slowly updated) |
| critic | Critic Q-network |
| critic_targ | Target critic network (slowly updated) |
| actor_opt | Adam optimizer for actor |
| critic_opt | Adam optimizer for critic |
| tau | soft update rate for target networks |
| gamma | discount factor (typically 0.99) |
| device | computation device (CPU or GPU) |

**Method: `augment_obs(obs, option)`**
- Concatenates one-hot encoded option to the state
- Input: observation (numpy array or tensor) and option index
- Output: augmented state [obs, one_hot(option)] of length obs_dim + num_options

**Method: `act(obs, noise_std=0.1, option=None)`**
- Selects action using the actor network
- Adds Gaussian noise for exploration (scaled by noise_std)
- Actions are clipped to valid range [-act_limit, act_limit]
- Returns action as numpy array of shape (act_dim,)

**Method: `update(replay_buffer, batch_size=256, update_iteration=1)`**
- Updates actor and critic networks from replay buffer samples
- For each update iteration:
  1. Sample random batch from replay buffer
  2. Compute target Q-value using target actor and target critic
  3. Update critic to minimize MSE(Q_predicted, Q_target)
  4. Update actor to maximize Q-value (policy gradient)
  5. Soft-update target networks: θ_target ← (1-τ)θ_target + τθ
- Returns critic and actor losses for monitoring


### class TD3

Twin Delayed DDPG (TD3) improves upon DDPG with three key modifications to reduce overestimation bias and improve stability.

**Key Differences from DDPG:**

1. **Twin Critics**: Two independent critic networks (Q₁ and Q₂); use minimum in target to reduce overestimation
2. **Delayed Policy Update**: Update actor less frequently than critics (every `policy_delay` critic updates) for better accuracy
3. **Target Policy Smoothing**: Add clipped noise to target action for smoother learning

**Class Attributes:**

| Attribute | Meaning  | 
|----------|----------|
| obs_dim | state dimension from environment | 
| act_dim | action dimension (continuous)  |
| act_limit | upper/lower bound for actions |
| num_options | number of discrete options (K) |
| obs_dim_aug | augmented state dimension = obs_dim + num_options |
| actor | Actor network |
| actor_targ | Target actor network |
| critic1, critic2 | Two critic Q-networks to reduce overestimation |
| critic1_targ, critic2_targ | Target critic networks |
| actor_opt | Adam optimizer for actor |
| critic1_opt, critic2_opt | Adam optimizers for critics |
| tau | soft update rate for target networks |
| gamma | discount factor (typically 0.99) |
| policy_noise | noise added to target action for smoothing |
| noise_clip | bounds for clipped noise |
| policy_delay | frequency of actor updates (e.g., update actor every 2 critic updates) |
| _n_critic_updates | counter to track when to update actor |
| device | computation device (CPU or GPU) |

**Method: `augment_obs(obs, option)`**
- Concatenates one-hot encoded option to the state
- Input: observation (numpy array or tensor) and option index
- Output: augmented state [obs, one_hot(option)] of length obs_dim + num_options

**Method: `act(obs, noise_std=0.1, option=None)`**
- Selects action using the actor network
- Adds Gaussian noise for exploration (scaled by noise_std)
- Actions are clipped to valid range [-act_limit, act_limit]
- Returns action as numpy array of shape (act_dim,)

**Method: `update(replay_buffer, batch_size=256, update_iteration=1)`**
- Updates actor and both critic networks from replay buffer samples
- For each update iteration:
  1. Sample random batch from replay buffer
  2. Add smoothing noise to target action and clip it
  3. Compute target Q-value using **minimum** of both target critics (reduces overestimation)
  4. Update both critics independently to minimize MSE loss
  5. **Every policy_delay updates**: Update actor using gradient from critic1
  6. Soft-update all target networks
- Returns critic losses, actor losses, and whether actor was updated
- This delayed update strategy allows critics to stabilize before actor moves

## Replay Buffer

### class ReplayBuffer

Buffer for storing and sampling reinforcement learning transitions in both standard RL (DDPG/TD3) 
and hierarchical RL (Option-Critic) settings.

Each stored transition includes the option and termination flag to support HRL training.

**Class Attributes:**

| Attribute | Meaning  | 
|----------|----------|
| storage | List of tuples, each containing one transition |
| max_size | Maximum capacity of the buffer (default: 1,000,000) |
| ptr | Pointer indicating where to write next when buffer is full (circular buffer) |

**Stored Transition Format:**

Each transition is a tuple: `(obs, next_obs, action, reward, done, option, terminated)`

- **obs**: np.array of shape (obs_dim,) - current observation/state
- **next_obs**: np.array of shape (obs_dim,) - next observation/state after action
- **action**: np.array of shape (act_dim,) - continuous action executed
- **reward**: float - reward received from environment
- **done**: float (0.0 or 1.0) - whether episode ended at this step
- **option**: int or None - the discrete option in use during this transition
- **terminated**: float (0.0 or 1.0) or None - whether the option terminated at this step due to β(s,o) > threshold

**Method: `push(obs, action, reward, next_obs, done, option=None, terminated=None)`**

Store a single transition in the buffer.

**Parameters:**
- `obs`: observation (np.array or list of shape (obs_dim,))
- `action`: action taken (np.array of shape (act_dim,))
- `reward`: scalar reward
- `next_obs`: resulting observation (np.array of shape (obs_dim,))
- `done`: episode termination flag (float or bool)
- `option`: (optional) discrete option index
- `terminated`: (optional) option termination flag

**Behavior:**
- If buffer is not full: appends new transition
- If buffer is full: overwrites at position `ptr` and advances `ptr` circularly
- All inputs are converted to appropriate dtypes (float32 for observations/actions/rewards)

**Method: `sample(batch_size)`**

Sample a random batch of transitions from the buffer for training.

**Returns (all as numpy arrays):**
- `state`: shape (B, obs_dim)
- `next_state`: shape (B, obs_dim)
- `action`: shape (B, act_dim)
- `reward`: shape (B, 1)
- `done`: shape (B, 1) - episode termination flags
- `option`: shape (B,) as int64, or None if no options were stored
- `terminated`: shape (B, 1) as float32, or None if no termination flags were stored

**Behavior:**
- Randomly selects `batch_size` transitions from storage
- Returns None for `option` if all sampled transitions have option = None
- Returns None for `terminated` if all sampled transitions have terminated = None
- All arrays properly shaped for batch processing in neural networks

**Method: `__len__()`**

Returns the current number of transitions in the buffer (not exceeding max_size).

---

## Key Hyperparameters and Configuration

### High-Level (Option-Critic) Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|----------------|
| `num_options` | Number of discrete options (K) | 4-8 |
| `tau` | Soft update rate for target networks | 0.005 |
| `eps_option` | Epsilon for epsilon-greedy option selection | 0.01-0.1 |
| `optv_lr` | Learning rate for OptionValue network | 1e-3 |
| `term_lr` | Learning rate for Termination network | 1e-4 |
| `delib_cost` | Penalty for switching options | 0.5 |
| `min_option_steps` | Minimum steps before option can terminate | 1 |
| `terminate_deterministic` | Use hard threshold (>0.5) vs stochastic termination | False |

### Low-Level (DDPG/TD3) Parameters

| Parameter | Description | DDPG | TD3 |
|-----------|-------------|------|-----|
| `gamma` | Discount factor | 0.99 | 0.99 |
| `tau` | Soft update rate | 0.005 | 0.005 |
| `actor_lr` | Actor learning rate | 1e-3 | 1e-3 |
| `critic_lr` | Critic learning rate | 1e-3 | 1e-3 |
| `hidden` | Hidden layer size | 256 | 256 |
| `policy_noise` | Target policy smoothing noise | - | 0.2 |
| `noise_clip` | Clipping range for noise | - | 0.5 |
| `policy_delay` | Delayed actor update frequency | - | 2 |

### Training Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|----------------|
| `batch_size` | Replay buffer batch size | 256 |
| `update_iteration` | Updates per environment step | 1-4 |
| `buffer_size` | Replay buffer capacity | 1,000,000 |
| `noise_std` | Exploration noise decay | 0.5 → 0.1 → 0.01 (over episodes) |

## Tips for Effective HRL Training

1. **Option Duration**: Set `min_option_steps` to enforce minimum commitment to each option. This prevents thrashing between options.

2. **Deliberation Cost**: Tune `delib_cost` to balance option switching vs. exploitation. Higher values discourage frequent switching.

3. **Learning Rate**: Set `term_lr` lower than `optv_lr` as termination learning can be noisier with small sample sizes.

4. **Algorithm Choice**: 
   - Use **DDPG** for simpler tasks or when sample efficiency is less critical
   - Use **TD3** for complex tasks requiring stable, sample-efficient learning

5. **Option Count**: Start with 4-8 options. Too few limits expressiveness; too many can slow learning.

6. **Monitoring**: Track:
   - `num_terminations`: Should increase steadily (options are terminating)
   - `num_option_switches`: Should show meaningful patterns, not random noise
   - Option value loss: Should decrease over time
   - Termination loss: Should stabilize (indicates learned termination behavior)






















