# HRL Toolkit

A modular Python toolkit for Hierarchical Reinforcement Learning with continuous control.

This project implements a two-level HRL architecture where a high-level controller selects discrete options and a low-level continuous controller executes actions conditioned on the selected option. The framework combines option-value learning, learnable termination functions, deliberation cost, and off-policy actor-critic algorithms such as DDPG and TD3.

The toolkit was developed as part of a research activity on Hierarchical Reinforcement Learning, with the goal of building a reusable experimental framework for studying temporal abstraction in continuous control environments.

---

## Overview

Standard reinforcement learning agents choose a primitive action at every time step. Hierarchical Reinforcement Learning introduces a higher level of decision-making through temporally extended behaviours called options.

In this project:

- the high-level controller selects an option;
- the low-level controller receives the state and selected option;
- the low-level controller outputs a continuous action;
- a termination function decides whether the current option should end;
- a deliberation cost discourages excessive option switching.

The main goal is to study whether this structure can produce meaningful option specialization and temporal segmentation in continuous control tasks.

---

## Main Features

- Hierarchical Reinforcement Learning with discrete options and continuous actions
- Option-value function \(Q_\Omega(s, o)\) for high-level option selection
- Learnable termination function \(\beta(s, o)\)
- Deliberation cost for more stable option durations
- Low-level continuous control with DDPG and TD3
- Option-conditioned actor and critic networks
- HRL-specific replay buffer
- Training logs, checkpoints, and annotated evaluation videos

---

## Method

The high-level controller estimates the value of each option through an option-value function:

```text
QΩ(s, o)
```

Options are selected using a greedy or epsilon-greedy rule.

The low-level policy is conditioned on both the environment state and the selected option:

```text
π(a | s, o)
```

The option is encoded and passed as an additional input to the actor and critic networks. This allows a single shared low-level controller to adapt its behaviour depending on the active option.

The termination function:

```text
β(s, o)
```

determines whether the current option should continue or terminate. A deliberation cost is applied when an option terminates, encouraging more persistent and meaningful temporal abstractions.

---

## Project Structure

```text
.
├── ddpg.py                    # DDPG low-level controller
├── td3.py                     # TD3 low-level controller
├── networks.py                # Neural network architectures
├── option_agent.py            # Main hierarchical agent
├── experience_replay.py       # Replay buffer
├── train_option_agent.py      # Training script
├── eval_annotated_video.py    # Annotated evaluation videos
├── utils.py                   # Utility functions
├── Documentation.md           # Additional documentation
└── README.md
```

---

## Experiments

The framework was tested on continuous control environments, mainly:

- `Pendulum-v1`
- `LunarLanderContinuous-v3`

`Pendulum-v1` was used as an initial sanity-check environment, while `LunarLanderContinuous-v3` provided richer dynamics for studying option specialization.

The experiments compare DDPG and TD3 as low-level controllers within the same hierarchical architecture.

The analysis focuses on:

- episode return;
- actor and critic losses;
- option-value loss;
- termination behaviour;
- option usage;
- effective option switches;
- qualitative inspection through annotated videos.

---

## References

- Sutton, Precup, and Singh (1999), *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*
- Bacon, Harb, and Precup (2017), *The Option-Critic Architecture*
- Harb et al. (2018), *When Waiting is not an Option: Learning Options with a Deliberation Cost*
- Lillicrap et al. (2015), *Continuous Control with Deep Reinforcement Learning*
- Fujimoto, Hoof, and Meger (2018), *Addressing Function Approximation Error in Actor-Critic Methods*
- Sutton and Barto (2018), *Reinforcement Learning: An Introduction*