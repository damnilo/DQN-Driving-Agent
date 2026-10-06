# DQN Driving Agent

A Deep Reinforcement Learning project that trains an autonomous driving agent with a **Double Dueling DQN** in [MetaDrive](https://github.com/metadriverse/metadrive). The agent learns lane following on fixed straight and curve maps, then fine-tunes on procedurally generated roads.

The project was developed as part of a university seminar on reinforcement learning and autonomous vehicle control.

---

## Features

* Double Dueling DQN in PyTorch, with a GRU over a 4-frame stack
* Prioritized experience replay and a soft-updated target network
* 28 discrete actions (7 steering values × 4 throttle values)
* Route-based reward and route waypoints, so lane changes do not cancel progress
* Staged training: behavior cloning, straight maps, curve maps, then procedural maps
* Block curriculum on procedural maps (straights and curves, then junctions, then the full MetaDrive mix)
* Held-out evaluation on seeds the trainer never uses
* CSV logs and checkpoints

---

## Observation

Each step builds one vector, then stacks the last 4 frames.

The vector starts with 16 waypoint features. Four points on the navigation route, at 5 m, 10 m, 20 m and 35 m, each contribute heading difference, curvature, and the point's position in the vehicle frame. The route follows navigation checkpoints, so at an intersection the waypoints point along the commanded turn.

The rest of the vector is:

* 9 ego features and 10 navigation features from MetaDrive's lidar observation
* Side and lane-line distances
* Navigation command, distances to the left and right lane boundaries, and lane-centre ratio
* The previous discrete action, normalised
* 240 lidar rays

Lidar is encoded by a 1D convolution and pooled to a single vector. The observation size is unchanged from the curve checkpoint, so `checkpoints/best_curve.pt` still loads.

---

## Action Space

Continuous controls are binned into 28 actions.

Steering:

```python
[-0.30, -0.18, -0.09, 0.00, 0.09, 0.18, 0.30]
```

Throttle:

```python
[-0.30, -0.05, 0.25, 0.60]
```

---

## Reward

Live reward follows metres travelled along the navigation route (`route_travelled`), clipped per step to [−1, 20] and scaled by 2. A lane change no longer resets the progress term. Heading alignment adds `cos(error) * 0.35`. Lateral offset and steering changes are penalised. There is a small per-step bonus of 0.05.

Terminal rewards:

| Outcome | Reward |
|---|---|
| Arrive at destination | +1000 |
| Crash or out of road | −100 |
| Horizon reached | −50 |
| Stuck | −40 |

Stuck means the vehicle moved less than 0.2 m in the world plane for more than 120 steps. The check uses world position, so a lane change is not treated as standing still.

---

## Training Pipeline

Run the stages in order. Each stage loads the checkpoint produced by the previous one.

```bash
python collect_idm.py
python -m train_bc
python -m train_straight
python -m train_curve
python -m train
```

`collect_idm.py` rolls out MetaDrive's IDM expert for 300 episodes: straight maps, then `SCSC` / `CSCS` / `CCCC`, then procedural maps. Transitions go to `dataset/expert_dataset.npz`. The stored action is the discrete bin that was executed.

`train_bc.py` imitates that dataset and writes `checkpoints/bc_pretrain_straight.pt`.

`train_straight.py` trains on `SSSS` (horizon 800) and saves `checkpoints/best_straight.pt` when success improves. Target success is 0.90.

`train_curve.py` loads the straight checkpoint and trains on `SCSC`, `CSCS`, and, after episode 800, `CCCC`. Horizons are 1000 and 1400. It saves `checkpoints/best_curve.pt`. Target success is 0.85.

`train.py` loads the curve weights only. The optimizer and step counter start fresh, and exploration starts at ε = 0.08 (decay to 0.02 over 300 000 steps). Training uses procedural maps of 4 blocks (`map=4`), horizon 2000, and seeds 0–49.

### Block curriculum

`map=4` samples four road blocks after the fixed first block. `train.py` changes which blocks can appear:

| Episodes | Blocks |
|---|---|
| 0–399 | Straight 0.35, Curve 0.65 |
| 400–999 | Straight 0.15, Curve 0.35, intersection 0.25, T-intersection 0.25 |
| 1000+ | MetaDrive default mix, including ramps and roundabouts |

Epsilon restarts when the block distribution changes. Every 40 episodes the run evaluates 10 greedy episodes on the full default mix, seeds 50–69, and saves `checkpoints/best_random.pt` when that success rate improves. Training stops at success 0.90 or at 4000 episodes. The same seeds can be drawn more than once in one evaluation, so the printed rate can double-count a map.

`EXPERT_RATIO_RANDOM` is 0. The existing expert file was labeled with the old reward. After a fresh `collect_idm.py`, raise it to 0.10–0.20 in `configs/env_config.py` to mix those transitions back in.

`python main.py` plays one greedy episode per map, loading `best_straight.pt`, `best_curve.pt`, or `best_random.pt`.

---

## Results

On the fixed curve maps the agent reaches the curve-stage target. The checkpoint from that stage is `checkpoints/best_curve.pt`.

Fine-tuning on procedural maps learned the straight-and-curve block mix first (about 78% of training episodes arrived in episodes 300–399), then junctions. Held-out success on the full block mix peaked at **0.80** on episode 599. That checkpoint is `checkpoints/best_random.pt`.

From episode 1000 the trainer switched to the full mix, including ramps and roundabouts. Training arrivals fell from about 50% to roughly 10–20%, and later held-out scores stayed near 0.30. The replay buffer had already dropped the earlier successful transitions, so the live weights at the end of that run are worse than `best_random.pt`. Use the saved checkpoint, not `final.pt` from that run.

Logs for the run are `logs/training_20261005_200429.csv` and `logs/evaluation_20261005_200429.csv`.

---

## Technologies

* Python 3.10
* PyTorch
* NumPy
* MetaDrive 0.4.3
* Gymnasium-style `reset` / `step` API

---

## Repository Structure

```text
├── agents/
│   ├── dqn_agent.py
│   ├── epsilon_scheduler.py
│   └── q_network.py
├── configs/
│   └── env_config.py
├── environment/
│   ├── action_mapper.py
│   ├── info_builder.py
│   ├── metadrive_env.py
│   ├── observation_builder.py
│   └── reward_function.py
├── replay/
│   └── expert_replay_buffer.py
├── training/
│   ├── checkpoint_manager.py
│   ├── curve_trainer.py
│   ├── evaluator.py
│   └── trainer.py
├── utils/
│   ├── action_discretizer.py
│   ├── frame_stack.py
│   └── logger.py
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

Developed by **Danilo Nikić** as part of a reinforcement learning seminar.
