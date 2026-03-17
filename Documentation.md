# HRL Toolkit Documentation

This Markdown document provides a structured overview of the Python modules
available in the workspace.  It is intended to help developers navigate the
hierarchical reinforcement learning (HRL) codebase, understand key classes and
functions, and see how components interact.

# Purpose of the code

# Workflow

The code is executed runnin the `train` function.

1. The envirionment is initialized 

2. The low level algorithm is select (Argument passed by the user in the configs), and the conseguent low level agent(`low level`) is initialized.
    -   Possible choice: DDPG, TD3  (Descrivi come funzionano e quali sono i parametri)

3. `agent` is create as an istance of the class `OptionAgent` --> (Dovrebbe essere la classe che si occupa di gestire le opzioni, quando cambiarle e come)
    - The `low level` algorithm is passed as one of the arguments of the `agent`.
    - Da quello che ho capito agent si occupa sia del low-level sia del high-level 

4.  Replay buffer (`buffer`) is initalized.
    - Spiega che tipo di informazioni salviamo nel replay buffer. (Datatype e contenuto)

5. First option and first state are selected randmoly. (chiedi conferma)

6. The training loop starts and it is repeated n = `total_steps` numbers of time: 

      7.  The agent is provided with the state(`obs`) and it is asked to select the best action and the best option .
           `action, option, did_terminate, term_steps = agent.act(obs, noise_std=noise_std, greedy_option=False)`

      8. The action result in a new state of the envirionent and an obtained reward (the reward is augmented with the deliberation cost to take in account costs associated to switching option).

      9. Chiarire meglio la questione terminazione episodio/opzione 

      10. The Replay Buffer is populated with ['obs (state of the env)', 'action', 'reward', 'next_obs', 'done(episode reached the goal(?))', 'option', 'terminate(option terminated(?))']

      11. We the buffer is populated enough --> agent.update()  --> Cosa fa??Quali pesi aggiorna? --> Non mi è chiaro cosa succede subito dopo. 

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

# High-Level

## OptionAgent

This script incorporate the logic of HRL above the continuous RL in low level (DDPG/TD3)
It's purpose is to choice a discrete option using an "options policy" and deciding when to change the option using a specific termination function.

Everything is handled by the class `OptionAgent`

| Attribute | Meaning  | 
|----------|----------|
| obs_dim | state dimension | 
| num_options| number of discrete options K|
| low_level_agent | low level RL agent | 
| device | | 
| tau |  | 
| hidden |  | 
| eps_option | optional epsilon-greedy on option selection |
| terminate_deterministic | if True, terminate when beta>0.5 instead of sampling Bernoulli | 
| min_option_steps|  | 
| optv_lr | | 
| term_lr |  | 




#### OptionValue
`OptionValue(input = obs_dim, output =  num_options, hidden=256)`

High-level Q-network producing a score for each discrete option, the score of each option is rapresent by the  Q value Q(s, o).
The network is composed as a classical feed forward nn.

It takes as input the state and produce as result what is the Q value of that state under different option. 

- How is the loss calulated?


#### Termination
`Termination(input = obs_dim, output =  num_options, hidden=256)`

This network calculate the termination function beta(s, o): probability of terminating option o in state s
The output will be K logits so each logit is how likely the respective option will end.

It takes as input the state and produce as result what is the the probability of terminating a specific option. 

- How is the loss calulated?

### Low-Level 

In the low level we implents classical RL algorithms.
Here the networks used to implements these algorithms. 

#### Actor
`Actor(obs_dim, act_dim, act_limit, hidden=256)`

- MLP mapping state inputs to bounded continuous actions.
- Two hidden layers with ReLU activation and layer normalization.
- Final `tanh` output scaled by `act_limit`.

#### Critic
`Critic(obs_dim, act_dim, hidden=256)`

- Estimates Q(s,a) values. Incorporates the action after the first hidden
  layer.


---

## ddpg.py

Deterministic policy gradient algorithm adapted for options.

- **Augmented states:** the current option is one-hot encoded and concatenated
  to the state vector.
- Both actor and critic take the augmented state as input.

### Important methods

- `augment_obs(obs, option)`: concatenates state and option encoding.
- `act(obs, noise_std=0.1, option=None)`: returns a (possibly noisy) action.
- `update(replay_buffer, batch_size=256, update_iteration=1)`: updates actor
  and critic using standard DDPG steps, then soft-updates target networks.
- `state_dict()` / `load_state_dict(sd)`: serialization helpers.

---

## td3.py

Twin Delayed DDPG with option-conditioning.

Key TD3 features:

1. **Double critics** (`critic1`, `critic2`); target uses the minimum Q-value.
2. **Delayed policy updates**: actor updated only every `policy_delay`
   critic steps.
3. **Target policy smoothing**: noise added to target actions.

Methods mirror `DDPG` but incorporate the above enhancements.
Serialization helpers include an actor-update flag.

---

## option_agent.py

High-level agent managing discrete options over a continuous low-level
controller (DDPG or TD3).


### Networks

- `option_value` (with target copy), estimates QΩ(s,o) for option selection.
- `termination`, predicting β(s,o) probability of terminating an option.

### State tracking

- `current_option`, `option_steps`, counts for terminations/switches.
- Exploration via `eps_option`; deliberation cost penalizes switching.

### Key methods

- `select_option(obs, greedy=False)`: epsilon-greedy or greedy option choice.
- `should_terminate(obs, option)`: sample termination or use deterministic
  threshold.
- `act(obs, noise_std=None, greedy_option=False)`: handles termination,
  switching, and delegates to low-level agent.
- `update(replay_buffer, batch_size, update_iteration)`: trains both the
  option-value and termination networks. Returns losses and low-level output.

---

## experience_replay.py

Simple replay buffer for RL or HRL with options.

- Stores tuples `(obs, next_obs, action, reward, done, option, terminated)`.
- `push(...)`: appends or overwrites samples in a circular buffer.
- `sample(batch_size)`: returns random minibatches, converting missing
  option/terminated values into `None`.

---

## eval_annotated_video.py

Script to run a trained agent and record annotated video.

- Overlays text (option, termination, step, reward) onto frames.
- Builds environment and agent based on command-line args.
- Loads checkpoint, runs episode, logs switches/terminations, writes video
  using `imageio`.

---

## train_option_agent.py

Training script for hierarchical option-based RL.

- Configures environment, agent (DDPG/TD3 + OptionAgent), replay buffer,
  and logging directories.
- Handles reward modification for deliberation cost.
- Updates low/high-level networks, logs per-episode statistics and saves
  configurations and checkpoints.


### Utilities

- `get_noise_std(ep)`: schedules exploration noise.
- `save_checkpoint(run_dir, agent, ...)`: saves snapshots.
- Command-line parser exposes numerous hyperparameters.

---

## Additional Notes

Files under `File Inutili/` are legacy training scripts and agents not required
for current usage.

This document is intended as a starting point.  For deeper understanding, refer
to inline comments and class docstrings within each file.
