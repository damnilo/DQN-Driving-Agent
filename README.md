# DQN Driving Agent

A Deep Reinforcement Learning project that trains an autonomous driving agent using the **Deep Q-Network (DQN)** algorithm. The agent learns to navigate procedurally generated road environments, avoid collisions, stay within lane boundaries, and successfully reach the destination through trial-and-error interaction with the environment.

---

## Project Overview

This project investigates the application of Deep Reinforcement Learning to autonomous driving tasks. A DQN agent is trained in a simulated driving environment where it receives observations from the environment and selects actions from a discrete action space consisting of steering and throttle combinations.

The primary objective is to learn a policy that maximizes long-term reward by:

* Following the road layout
* Maintaining forward progress
* Avoiding collisions
* Staying on the road
* Successfully completing routes

The project was developed as part of a university seminar work focused on Reinforcement Learning and autonomous vehicle control.

---

## Features

* Double Dueling Deep Q-Network (DQN) implementation using PyTorch
* Prioritized Experience Replay
* Target Network stabilization
* Epsilon-Greedy exploration strategy
* Reward shaping for driving behavior
* Training on procedurally generated road maps
* Evaluation on unseen environments
* Logging of training statistics
* Model checkpointing and recovery

---

## State Representation

The agent receives a vector of observations describing the current driving state, including information such as:

* Vehicle speed
* Heading error
* Lateral position
* Road geometry information
* LiDAR sensor rays
* Additional environment-specific features

The observation space is designed to provide sufficient information for lane following and navigation decisions.

---

## Action Space

The continuous vehicle controls are discretized into a finite action space.

### Steering Values

```python
[-0.30, -0.18, -0.09, 0.00, 0.09, 0.18, 0.30]
```

### Throttle Values

```python
[-0.30, -0.05, 0.25, 0.60]
```

Each action represents a unique combination of steering and throttle values.

---

## DQN Architecture

The neural network approximates the action-value function:

```math
Q(s,a)
```

The network consists of:

* 1D Convolutional encoder for LiDAR sensors
* Linear projection for non-LiDAR information
* GRU over the stacked-frame sequence
* Seperate Value and Advantage heads for Double Dueling DQN

The target network is periodically updated to improve training stability.

---

## Reward Function

The reward function encourages safe and efficient driving behavior.

Positive rewards are given for:

* Forward progress along the route
* Maintaining road position
* Completing the route

Negative rewards are applied for:

* Collisions
* Driving off the road
* Excessive inactivity
* Unstable driving behavior

Reward shaping is used to accelerate learning and improve convergence.

---

## Training Process

The training loop follows the standard DQN procedure:

1. Observe current state.
2. Select action using epsilon-greedy policy.
3. Execute action in the environment.
4. Store transition in replay buffer.
5. Sample mini-batches from replay memory.
6. Update online network.
7. Periodically synchronize target network.
8. Decay exploration rate.

---

## Results

The agent was evaluated on both training maps and previously unseen maps.

Performance metrics include:

* Episode reward
* Success rate
* Collision rate
* Off-road rate
* Average episode length

The trained agent demonstrated the ability to generalize to new road configurations while maintaining a high route completion rate.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* OpenAI Gym-style environment
* Matplotlib
* Reinforcement Learning

---

## Repository Structure

```text
├── agent/
│   ├── dqn_agent.py
│   ├── epsiolon_scheduler.py
│   └── q_network.py
│   
├── environment/
│   ├── action_mapper.py
│   ├── info_builder.py
│   ├── metadrive_env.py
│   ├── observation_builder.py
│   └── reward_function.py
│  
├── replay/
│   └── expert_replay_buffer.py
│   
├── training/
│   ├── checkpoint_manager.py
│   ├── curve_trainer.py
│   ├── evaluator.py
│   └── trainer.py
│   
├── utils/
│   ├── action_discretizer.py
│   ├── frame_stack.py
│   └── logger.py
│
├── collect_idm.py
├── main.py
├── train.py
├── train_bc.py
├── train_straight.py
├── train_curve.py
└── README.md
```

---

## Author

Developed by **Danilo Nikić** as part of a Reinforcement Learning research and educational project.

