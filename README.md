# LEAP Hand Dexterous Grasping

A reinforcement learning pipeline for training the **LEAP Hand** robotic manipulator to grasp and lift a cube in MuJoCo, using PPO (Proximal Policy Optimization) from Stable Baselines 3.

**Final Result: 85% true success rate over 100 randomized deterministic evaluation episodes.**

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
│   ├── final_model.zip         # Final saved model
│   ├── vecnormalize.pkl        # Observation/reward normalization stats
│   ├── eval_results.txt        # Final evaluation metrics
│   └── learning_curve.png      # Training reward curve plot
├── videos/
│   └── grasp_demo_ep*.mp4      # Demonstration recordings
├── train.py                    # PPO training script (2M timesteps)
├── eval.py                     # Evaluation script (100 episodes)
├── resume_train.py             # Resume training from checkpoint
├── record_video.py             # Record demonstration videos
├── plot_learning_curve.py      # Generate learning curve plot
├── visualize.py                # Live MuJoCo viewer
├── generate_report.py          # Generates the PDF technical report
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
Trains for 2,000,000 timesteps. Checkpoints are saved every 50,000 steps to `results/checkpoints/`. TensorBoard logs are written to `results/tensorboard/`.

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
**Success criterion:** cube `Z > 0.090 m` for 20 consecutive steps.

### Record demonstration videos
```bash
python3 record_video.py
```
Saves episode recordings to `videos/`.

### Visualize the policy live
```bash
python3 visualize.py
```
Opens an interactive MuJoCo viewer running the policy in real time.

### Generate the PDF report
```bash
python3 generate_report.py
```
Generates `report.pdf` — the 2-page technical summary required for Task A.

### Plot the learning curve
```bash
python3 plot_learning_curve.py
```
Parses TensorBoard event files and saves `results/learning_curve.png`.

---

## Final Evaluation Results

After ~3M total training timesteps (2M fresh + 1M resume):

| Metric | Value |
|---|---|
| Mean Episode Reward | 2846.96 ± 3305.17 |
| Mean Episode Length | 994.01 / 1000 steps |
| **True Success Rate** | **85% (85/100 episodes)** |
| Z threshold | 0.090 m (sustained 20 consecutive steps) |
| Total Training Steps | ~3,000,000 |

---

## Environment Design

### Observation Space (39 values)
| Component | Dim | Description |
|---|---|---|
| Joint positions | 16 | Current angles of all 16 LEAP Hand actuators (rad) |
| Joint velocities | 16 | Angular velocity of each joint (rad/s) |
| Object position | 3 | Cube centroid in world XYZ (m) |
| Object quaternion | 4 | Cube orientation as unit quaternion [w, x, y, z] |

### Action Space (16 values)
Continuous `Box[-1, 1]^16` — applied as **delta joint angle targets**:
```
ctrl[t+1] = clip(ctrl[t] + action × 0.05,  joint_low,  joint_high)
```
`max_delta = 0.05 rad/step` enforces smooth, physically realistic finger movements.

### Reward Function (Final Design)
The final converged reward function that achieved 85% success:

| Component | Formula | Purpose |
|---|---|---|
| Contact reward | `+0.5 × min(n_hand_contacts, 4)` | Reward stable multi-finger grip |
| Lift reward | `+10.0/step` if `Z > 0.085m` AND grasping | Continuous bonus for sustained lift |
| Corner airborne | `+3.0 × max(0, min_corner_Z − 0.055)` | All 8 cube corners must stay above table |
| XY drift penalty | `−15.0 × max(0, drift − 0.03)` | Prevent lateral pushing |
| Drop penalty | `−20.0` when cube falls below threshold after lift | Punish grip instability |
| Spawn-Z tether | `−20.0 × max(0, 0.085 − obj_Z)` | Prevent cube from being dragged down |
| Time penalty | `−0.5/step` | Force the agent to act, not idle |
| Action penalty | `−0.01 × Σ(action²)` | Reduce jitter |

**Key design insight:** The approach reward was intentionally **removed** from the final design. Although useful early in training for guiding fingers toward the cube, it caused lateral contact forces that pushed the cube out of reach once the policy matured. After sufficient training (~2M steps), it became counterproductive.

### Physics
- **Substeps**: 5 MuJoCo substeps per policy step at `timestep=0.002s`
- **Hand orientation**: Top-down (`quat="1 0 0 0"`), palm at Z=0.16m, to oppose gravity during lifting
- **Cube spawn**: `Z=0.085m` (elevated at success threshold to eliminate scooping forces)
- **OOB termination**: `dist_xy > 0.20m` or `Z < 0.0m`

---

## Stack

- **Simulator**: MuJoCo 3.x
- **RL Library**: Stable Baselines 3 (PPO)
- **Hand Model**: LEAP Hand from `mujoco_menagerie` — 16 joints, 16 actuators
- **Python**: 3.12

---

## Key Technical Decisions

1. **Top-down orientation**: A fixed-base hand cannot generate upward Z-force in a sideways configuration. Re-orienting the palm downward was essential for physically valid lifting.
2. **Delta control (`max_delta=0.05`)**: Absolute position control caused infinite-acceleration exploits. Delta control with a small max step enforces physically realistic, smooth finger movements.
3. **Zero reward below threshold**: The decisive fix. Any positive lift reward below `Z=0.085m` (even 0.001 points/step) gives the agent a profitable "hovering" strategy. Setting it strictly to zero forced the agent to commit to a full lift.
4. **Time penalty (`-0.5/step`)**: Without this, the agent was content to idle. The time penalty makes doing nothing worse than any failed grasp attempt, forcing active exploration.
5. **Cube spawned at threshold height**: Spawning the cube at `Z=0.085m` (the success threshold) means the fingers only need to **close**, not scoop upward. Scooping creates lateral forces that push the cube out of reach.

---

## Reward Design Evolution (What Worked / Failed)

| Run | Key Change | Result |
|---|---|---|
| Run 1 | Velocity penalty | ❌ Highly unstable (contact spikes in `qvel`) |
| Run 2 | Action penalty + absolute control | ❌ Violent flinging / hover exploit |
| Run 3 | Delta control + true contact filter | ✅ 15% true success |
| Run 4 | Linear LR + sustain=25 + OOB widened | ❌ 0% (reward hacking re-emerged) |
| Run 5 | **Zero below-threshold + time penalty + Z=0.085m threshold** | ✅ **85% true success** |
