# LEAP Hand Dexterous Grasping

A reinforcement learning pipeline for training the **LEAP Hand** robotic manipulator to grasp and lift a cube in MuJoCo, using PPO (Proximal Policy Optimization) from Stable Baselines 3.

**Final Result: 15% true success rate over 100 randomized evaluation episodes. (Proxy metric previously reported 96%).**

---

## Project Structure

```
leap-dexterous-grasping/
├── env/
│   └── leap_grasp_env.py       # Custom Gymnasium environment
├── mujoco_menagerie/leap_hand/
│   ├── grasp_scene.xml         # MuJoCo scene (table + cube + hand)
│   └── right_hand_grasp.xml   # LEAP Hand model (top-down orientation)
├── configs/
│   └── ppo_config.yaml         # All training hyperparameters
├── results/
│   ├── best_model/             # Best checkpoint during training
│   ├── eval_results.txt        # Final evaluation metrics
│   └── learning_curve.png      # Training reward curve plot
├── videos/
│   └── grasp_demo_ep*.mp4      # 5 demonstration recordings
├── train.py                    # PPO training script
├── eval.py                     # Evaluation script (100 episodes)
├── resume_train.py             # Resume training from checkpoint
├── record_video.py             # Record demonstration videos
├── plot_learning_curve.py      # Generate learning curve plot
├── visualize.py                # Live MuJoCo viewer
├── TRAINING_LOG.md             # Reward tuning & debugging log
└── requirements.txt            # Frozen dependencies
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd leap-dexterous-grasping

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train from scratch
```bash
python3 train.py
```
Trains for 1,500,000 timesteps by default. Checkpoints are saved every 50,000 steps to `results/checkpoints/`. TensorBoard logs are written to `results/tensorboard/`.

### Resume training from a checkpoint
```bash
python3 resume_train.py
```
Loads `results/best_model/best_model.zip` and its normalization stats, then continues training for an additional 1,000,000 steps.

### Monitor training with TensorBoard
```bash
tensorboard --logdir results/tensorboard/
# Open http://localhost:6006
```

### Evaluate the trained policy
```bash
python3 eval.py
```
Runs 100 deterministic evaluation episodes and prints metrics to stdout and `results/eval_results.txt`.

### Record demonstration videos
```bash
python3 record_video.py
```
Saves 5 episode recordings to `videos/grasp_demo_ep*.mp4`.

### Visualize the policy live
```bash
python3 visualize.py
```
Opens an interactive MuJoCo viewer running the policy in real time.

### Plot the learning curve
```bash
python3 plot_learning_curve.py
```
Parses TensorBoard event files and saves `results/learning_curve.png`.

---

## Expected Results

After 2.5M total training timesteps (1.5M fresh + 1M resume):

| Metric | Value |
|---|---|
| Mean Episode Reward | 1398.06 ± 537.60 |
| Mean Episode Length | 414.93 / 500 steps |
| **True Success Rate** | **15% (15/100 episodes)** |

### The Success Metric Flaw
The original rigorous physical success criterion demanded the object be lifted to `Z > 0.25m`. However, this was discovered to be **physically impossible**, as the LEAP Hand's palm is statically mounted at `Z = 0.16m`. The cube cannot be lifted to 0.25m from below the palm.

When corrected to a physically possible threshold of `Z > 0.10m` (a 2.5cm sustained lift off the table for 10 steps), the policy achieves a **15% true success rate**.

The previously reported 96% was the result of a flawed proxy metric (`ep_reward > 500` + early termination). The agent learned to "hack" the dense reward function by hovering the cube just below the threshold (e.g., `Z=0.09m`) for the entire 500-step episode, avoiding the risk of dropping it while accumulating massive reward.

This repository serves as a case study in the dangers of dense reward shaping, proxy evaluation metrics, and failing to verify physical kinematic constraints in Reinforcement Learning.

---

## Environment Design

- **Observation space**: 39 values — 16 joint positions + 16 joint velocities + 3 object position + 4 object quaternion
- **Action space**: 16 continuous values in `[-1, 1]`, applied as delta joint angle targets (`max_delta = 0.1 rad/step`)
- **Reward**: `approach_reward + contact_reward + lift_reward - action_penalty`
- **Physics**: 5 MuJoCo substeps per policy step at `timestep=0.002s`
- **Hand orientation**: Top-down (`quat="1 0 0 0"`), palm at Z=0.16m, to oppose gravity during lifting

---

## Stack

- **Simulator**: MuJoCo 3.1.6
- **RL Library**: Stable Baselines 3 2.3.2 (PPO)
- **Hand Model**: LEAP Hand from `mujoco_menagerie` — 16 joints, 16 actuators
- **Python**: 3.12.3

---

## Key Technical Decisions

1. **Top-down orientation**: A fixed-base hand cannot generate upward Z-force in a sideways configuration. Re-orienting the palm downward was essential for physically valid lifting.
2. **Delta control**: Absolute position control caused infinite-acceleration exploits. Delta control with `max_delta=0.1 rad/step` enforces physically realistic finger movements.
3. **True contact detection**: Filtered out table/floor contacts so the agent is only rewarded for hand-on-cube contact, preventing the "hover near table" reward hack.
