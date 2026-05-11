# Training and Reward Tuning Log

This log documents the iterative reward engineering process, policy exploits discovered, and design decisions made during training of the LEAP Hand grasping policy.

---

## Run 1 — The "Jitter" Problem

**Setup:** PPO with Approach (+), Contact (+), Lift (+), Velocity Penalty (−).

**Observed behaviour:** Agent froze completely or flailed violently.

**Root cause:** High-frequency physics jitter from MuJoCo contacts caused massive, unpredictable spikes in `qvel`. The velocity penalty heavily punished the agent any time it touched the cube, making contact avoidance the optimal strategy.

**Fix:** Remove velocity penalty entirely. Never use instantaneous `qvel` as a reward signal in contact-rich MuJoCo environments.

---

## Run 2 — Action Penalties & the "Hovering" Hack

**Changes:** Replaced velocity penalty with action regularisation (`−0.001 × Σctrl²`).

**Observed behaviour:** Violent movements persisted (cube flinging), then the agent learned to hover its fingers directly above the cube without touching it.

**Root causes (two separate issues):**

1. **Infinite acceleration via absolute position control.** The network learned to command a joint from fully open to fully closed in one timestep (0.01 s), causing physically unrealistic flinging. Absolute position control was incompatible with stable simulation.

2. **Table-contact reward hacking.** The contact reward counted *all* contacts involving the cube. Because the cube rests on the table, it inherently has 4 contact points at spawn. The agent learned that touching the cube could tilt it and break those 4 free contacts — so the optimal strategy was to hover nearby and farm the approach reward without disturbing the cube.

**Fixes applied in Run 3:** (a) delta control, (b) filtered contact reward.

---

## Run 3 — Delta Control & True Contact Detection

**Changes:**
1. **Delta control:** Action space `[−1, 1]`, applied as `ctrl[t+1] = ctrl[t] + action × max_delta`. Enforces smooth movements, caps effective joint velocity.
2. **True contact filter:** Explicitly excluded contacts involving `table_top` and `floor` geoms. Agent rewarded only for hand–cube contacts.
3. **OOB termination:** `dist_xy > 0.20 m` or `Z < 0.0 m` triggers early termination with `−50` penalty.

**Result:** Policy began grasping. Initial success rate ~15%. The hovering exploit was eliminated but a new one emerged (see Run 4).

---

## Run 4 — The "Hovering Below Threshold" Exploit

**Problem discovered after Run 3:** Any positive lift reward below `Z = 0.085 m` made hovering at `Z = 0.084 m` worth ~2800 pts/episode — more profitable than risking a full lift. The agent learned to hold the cube as high as possible *without* crossing the threshold.

**Root cause:** Continuous lift reward (proportional to height) created a smooth gradient that the agent exploited near-but-below the threshold.

**Fix applied in Run 5:** Set lift reward to **strictly zero** below `Z = 0.085 m`. The reward function is:
```
r_lift = 10.0/step  if  obj_Z > 0.085 m  AND  n_contacts >= 2
r_lift = 0.0        otherwise
```
Combined with time penalty (−0.5/step), hovering became net-negative.

---

## Run 5 (Final) — Full Convergence at 85% Success

**All changes combined:**
- Zero lift reward below threshold
- Spawn-Z tether: `−20 × max(0, 0.085 − obj_Z)` — continuous gradient toward threshold
- Corner airborne reward: `+3.0 × max(0, min_corner_Z − 0.055)` — verifies full lift
- XY drift penalty: `−15 × max(0, |xy_drift| − 0.03)` — prevents lateral pushing
- Drop penalty: `−20` one-shot when cube falls below threshold after lift
- Time penalty: `−0.5/step`
- Action penalty: `−0.01 × Σ(action²)`
- Cube spawned at `Z = 0.085 m` (success threshold) to eliminate scooping forces

**Two-phase training:**
- **Phase 1 (0–2.677 M steps, `train.py`):** Approach reward active (`+10 × max(0, 0.2 − mean_dist)`). Agent learns to find the cube.
- **Phase 2 (2.677 M–3.677 M steps, `resume_train.py`):** Approach reward manually set to `0.0` in `leap_grasp_env.py` line 100 before resuming. By 2 M steps the approach reward was counterproductive — it drove all 4 fingertips inward simultaneously, creating lateral forces that ejected the cube. This is a **manual curriculum change**, not automated. To reproduce: edit line 100 of `env/leap_grasp_env.py` to set `approach_reward = 0.0`, then run `resume_train.py`.

**Final result:** 85% success rate over 100 deterministic evaluation episodes (Z > 0.090 m, sustained 20 steps).

---

## Post-Training Analysis — Honest Limitations

### Reward arithmetic reconciliation
Mean episode reward 2846 ± 3305 reflects a **bimodal distribution**:
- Success episodes (~3350 mean): Cube is lifted above threshold for ~200–250 steps (not the full episode). Breakdown: lift reward ≈ 10 × 200 = 2000, contact ≈ 2 × 994 = 1988, time penalty = −0.5 × 994 = −497, other ≈ −100. Total ≈ 3391 ✓
- Failure episodes (~−300 mean): No lift; only contact reward minus time and tether penalties.

The policy satisfies the **binary** 20-step success criterion but does **not** consistently hold the cube for the full episode — a meaningful distinction.

### Spawn shortcut
Spawning at `Z = 0.085 m` (= success threshold) means the policy never learned to lift from a resting surface. It learned only to grip and hold in place. In a real deployment where the cube starts on the table (`Z = 0.075 m`), this policy would fail. A curriculum gradually lowering spawn height would fix this.

### Grasp quality
- Contact reward counts geometry contacts (not force-closure quality). A light touch scores identically to a firm grip.
- The 1–2 second hold duration traces to: no palm load-bearing surface (top-down, no wrist), action penalty suppressing corrective grip force, and the success criterion requiring only 20 steps.

### norm_obs=False
Observation normalization was deliberately disabled. All 39 observation dimensions are naturally bounded (joint limits, delta-control velocity cap, fixed workspace). Quaternion running-stats normalization would distort geometric meaning. The two-phase curriculum also shifts the observation distribution, making running stats from Phase 1 harmful in Phase 2. Reward normalization was kept (`clip_reward=10.0`) because raw rewards span a wide range.

### Learning curve annotation (corrected)
The old learning curve showed "96% Success Rate (1632 mean reward)" as a mid-training annotation. This was misleading — it came from the `EvalCallback` using 5 episodes and VecNormalize-scaled rewards, not from a proper deterministic evaluation. The correct final result is 85% on 100 raw-reward deterministic episodes. The new `plot_learning_curve.py` reads directly from monitor CSVs and `evaluations.npz` and shows both panels (reward + success rate) honestly.
