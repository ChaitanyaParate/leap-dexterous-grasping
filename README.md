# LEAP Hand Dexterous Grasping

A reinforcement learning pipeline for training the **LEAP Hand** robotic manipulator to grasp and lift a cube in MuJoCo, using PPO (Proximal Policy Optimization) from Stable Baselines 3.

**Final result: 85% true success rate over 100 randomised deterministic evaluation episodes.**  
Success criterion: cube `Z > 0.090 m` sustained for 20 consecutive steps.

> **Honest caveat:** The policy satisfies the binary success criterion but does not consistently hold the cube for the full episode (~200–250 lift-steps out of 1000). Grasp quality is marginal — see [Known Limitations](#known-limitations) for a full analysis.

---

## Project Structure

```
leap-dexterous-grasping/
├── env/
│   └── leap_grasp_env.py       # Custom Gymnasium environment (reward, obs, action)
├── mujoco_menagerie/leap_hand/
│   ├── grasp_scene.xml         # MuJoCo scene (table + cube + hand + camera)
│   └── right_hand_grasp.xml   # LEAP Hand model (top-down orientation)
├── configs/
│   └── ppo_config.yaml         # All training hyperparameters (reference only)
├── results/
│   ├── best_model/             # Best checkpoint saved during training
│   ├── final_model.zip         # Final trained model (after resume)
│   ├── vecnormalize.pkl        # VecNormalize stats (required for eval/video)
│   ├── eval_results.txt        # Final evaluation metrics (100 episodes)
│   ├── learning_curve.png      # Two-panel learning curve (reward + success rate)
│   └── logs/
│       ├── monitor.csv                     # Per-episode reward log (fresh run)
│       ├── monitor_resume.csv.monitor.csv  # Per-episode reward log (resume run)
│       └── evaluations.npz                 # EvalCallback checkpoints (resume phase)
├── videos/
│   └── grasp_demo_ep*.mp4      # Demonstration recordings (5 episodes)
├── train.py                    # PPO training script — Phase 1 (0–2.677 M steps)
├── eval.py                     # Evaluation script — 100 deterministic episodes
├── resume_train.py             # Resume training — Phase 2 (2.677–3.677 M steps)
├── record_video.py             # Record demonstration videos (5 episodes)
├── plot_learning_curve.py      # Generate 2-panel learning curve from CSV/npz data
├── visualize.py                # Live MuJoCo viewer (interactive)
├── TRAINING_LOG.md             # Detailed reward engineering & debugging log
└── requirements.txt            # Frozen Python dependencies
```

---

## Installation

```bash
git clone <repo-url>
cd leap-dexterous-grasping

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Train from scratch (Phase 1)
```bash
python3 train.py
```
Trains for ~2.677 M timesteps. Saves checkpoints every 50 k steps to `results/checkpoints/`.  
**Before running Phase 2:** set `approach_reward = 0.0` in `env/leap_grasp_env.py` line 100.

### Resume training (Phase 2)
```bash
python3 resume_train.py
```
Loads the checkpoint from Phase 1 and continues for 1 M steps with approach reward disabled.  
The two-phase split is a **manual curriculum** — see `TRAINING_LOG.md` for the full procedure.

### Evaluate the trained policy
```bash
python3 eval.py
```
Runs 100 deterministic episodes. Prints per-episode results and writes `results/eval_results.txt`.  
Success criterion: cube `Z > 0.090 m` for 20 consecutive steps.

### Record demonstration videos
```bash
python3 record_video.py
```
Records 5 episodes to `videos/grasp_demo_ep*.mp4` using the trained policy.

### Visualize live
```bash
python3 visualize.py
```
Opens an interactive MuJoCo viewer running the policy in real time.

### Plot the learning curve
```bash
python3 plot_learning_curve.py
```
Reads `results/logs/monitor.csv`, `monitor_resume.csv.monitor.csv`, and `results/logs/evaluations.npz`.  
Saves a 2-panel figure (episode reward + success rate) to `results/learning_curve.png`.

### Monitor training with TensorBoard
```bash
tensorboard --logdir results/tensorboard/
```

---

## Final Evaluation Results

| Metric | Value |
|---|---|
| Mean Episode Reward | 2846.96 ± 3305.17 |
| Mean Episode Length | 994.01 / 1000 steps |
| **True Success Rate** | **85% (85 / 100 episodes)** |
| Z threshold | 0.090 m (sustained 20 consecutive steps) |
| Total Training Steps | ~3,677,000 (2.677 M + 1 M resume) |
| Eval Policy | Deterministic (no exploration noise) |

**Reward variance note:** The large std (3305 > mean 2846) reflects a bimodal distribution: success episodes average ~3350, failure episodes average ~−300. This is structural, not random — see [Known Limitations](#known-limitations).

---

## Environment Design

### Observation Space (39 values)
| Component | Dim | Description |
|---|---|---|
| Joint positions | 16 | Current angles of all 16 actuators (rad) |
| Joint velocities | 16 | Angular velocity of each joint (rad/s) |
| Object position | 3 | Cube centroid in world XYZ (m) |
| Object quaternion | 4 | Cube orientation [w, x, y, z] |

**16 actuators map to 4 fingers × 4 joints:**
- Index: `if_mcp, if_rot, if_pip, if_dip`
- Middle: `mf_mcp, mf_rot, mf_pip, mf_dip`
- Ring: `rf_mcp, rf_rot, rf_pip, rf_dip`
- Thumb: `th_cmc, th_axl, th_mcp, th_ipl`

### Action Space (16 values)
Continuous `Box[−1, 1]^16` — applied as delta joint angle targets:
```
ctrl[t+1] = clip(ctrl[t] + action × 0.05,  joint_low,  joint_high)
```
`max_delta = 0.05 rad/step` caps effective joint velocity at 5 rad/s (50 Hz control).

### Reward Function (Final Design)
| Component | Formula | Purpose |
|---|---|---|
| Contact | `+0.5 × min(n_hand_contacts, 4)` | Hand–cube contacts only (table/floor excluded) |
| Lift | `+10.0/step` if `Z > 0.085 m` AND grasping, else `0.0` | Zero below threshold eliminates hovering exploit |
| Corner airborne | `+3.0 × max(0, min_corner_Z − 0.055)` | All 8 cube corners above table surface |
| XY drift | `−15.0 × max(0, drift − 0.03)` | Prevent lateral pushing |
| Drop | `−20.0` one-shot when cube falls after lift | Punish grip loss |
| Spawn-Z tether | `−20.0 × max(0, 0.085 − obj_Z)` | Continuous gradient toward threshold |
| Time | `−0.5/step` | Makes hovering net-negative |
| Action | `−0.01 × Σ(action²)` | Reduce jitter |

**Approach reward** (`+10 × max(0, 0.2 − mean_dist)`) was active in Phase 1 only. It was manually removed before Phase 2 because it caused lateral contact forces that ejected the cube once the policy matured.

**Penalty interaction:** The spawn-Z tether and zero lift reward are not redundant — they serve different roles. The tether provides a *continuous gradient* pulling the cube upward below threshold. The zero lift reward is a *sparse gate* removing the hovering signal. Together they form a potential well toward Z = 0.085 m.

### Physics & Setup
- **Substeps:** 5 MuJoCo substeps per policy step (`timestep = 0.002 s` → 50 Hz control)
- **Hand orientation:** Top-down (`quat="1 0 0 0"`), palm at Z = 0.16 m
- **Cube spawn:** `Z = 0.085 m`, XY randomised ±1 cm
- **OOB termination:** `dist_xy > 0.20 m` or `Z < 0.0 m` → early termination, −50 penalty
- **VecNormalize:** `norm_obs=False, norm_reward=True, clip_reward=10.0`

---

## Key Technical Decisions

1. **Top-down orientation:** A fixed-base hand without a wrist can only generate upward Z-force when the palm faces down. Side or angled configurations could not oppose gravity.

2. **Delta control (`max_delta = 0.05`):** Absolute position control caused infinite-acceleration exploits — a joint could traverse its full range in one timestep. Delta control caps effective velocity and enforces smooth movements.

3. **Zero reward below threshold:** Any positive reward near Z = 0.085 m created a "hover just below" exploit. Setting lift reward strictly to zero forced the agent to commit to crossing the threshold.

4. **Time penalty (−0.5/step):** Makes every timestep of failure costly. Without it, the agent had no incentive to act. Combined with zero lift reward below threshold, hovering becomes net-negative.

5. **Cube at spawn height = threshold:** Placing the cube at Z = 0.085 m eliminated the scooping motion (finger sweep upward from table height), which consistently produced lateral forces. Fingers only need to close, not sweep.

6. **norm_obs=False:** All 39 observations are naturally bounded. Running-stats normalization of quaternion components distorts geometric meaning. The curriculum shift (approach reward removal) also invalidates Phase 1 running stats for Phase 2. Only rewards are normalised.

---

## Reward Engineering History

| Run | Key Change | Outcome |
|---|---|---|
| 1 | Velocity penalty | ❌ Untunable — qvel spikes from contacts |
| 2 | Action penalty + absolute control | ❌ Flinging + table-contact hover exploit |
| 3 | Delta control + true contact filter | ⚠️ ~15% success — below-threshold hover emerged |
| 4 | Continuous lift height reward | ❌ Hovering at Z=0.084m more profitable than full lift |
| 5 | **Zero below threshold + time penalty + spawn at Z=0.085m** | ✅ **85% success** |

See `TRAINING_LOG.md` for full root-cause analysis of each run.

---

## Known Limitations

### 1. Binary success ≠ sustained hold
The policy achieves the 20-step criterion but holds the cube for only ~200–250 steps (4–5 seconds) before it slips. The arithmetic: lift reward (10 × 200 = 2000) + contact (2 × 994 = 1988) − time (−497) ≈ 3391 matches the observed ~3350 success mean. **The policy does not demonstrate sustained robust grasping.**

### 2. Spawn height shortcut
The cube spawns at the success threshold (`Z = 0.085 m`), not on the table (`Z = 0.075 m`). The policy never learned to lift from a resting surface — only to grip and hold. It would fail in any real deployment where the cube starts lower.

### 3. Marginal grasp quality
- Contact reward counts contacts, not force-closure quality
- No pre-grasp approach phase (cube appears already at threshold height)
- Policy memorised one narrow hand-object geometry (±1 cm spawn noise)
- No palm load-bearing surface in top-down configuration

### 4. Reward variance
Mean reward 2846 ± 3305 reflects bimodal outcomes. High std is structural: 85 success episodes (~3350 each) vs. 15 failure episodes (~−300 each).

### What would improve this
- Hold-duration reward (+r per step beyond the first 20) to incentivise sustained grasping
- Force-closure quality metric instead of contact count
- Curriculum lowering spawn height from 0.085 m → 0.075 m over training
- Wider spawn randomisation (±3 cm, ±5° rotation) to prevent geometry memorisation

---

## Stack

- **Simulator:** MuJoCo 3.x
- **RL Library:** Stable Baselines 3 — PPO
- **Hand Model:** LEAP Hand from `mujoco_menagerie` — 4 fingers, 16 joints, 16 actuators
- **Python:** 3.12
