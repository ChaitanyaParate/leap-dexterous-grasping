import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('mujoco_menagerie/leap_hand/grasp_scene.xml')
d = mujoco.MjData(m)

mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
d.ctrl[:] = m.actuator_ctrlrange[:, 1] # Close fingers to grasp

vel_penalties = []
for i in range(500):
    mujoco.mj_step(m, d)
    if i > 100: # After initial snapping
        vel_sq = np.sum(np.square(d.qvel[:16]))
        vel_penalties.append(-0.0005 * vel_sq)

print(f"Average velocity penalty while grasping (steps 100-500): {np.mean(vel_penalties):.4f}")

# Now try opening fingers fully (no contact with anything)
mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
d.ctrl[:] = m.actuator_ctrlrange[:, 0] # Open fingers fully

vel_penalties = []
for i in range(500):
    mujoco.mj_step(m, d)
    if i > 100: # After initial snapping
        vel_sq = np.sum(np.square(d.qvel[:16]))
        vel_penalties.append(-0.0005 * vel_sq)

print(f"Average velocity penalty while fully open (steps 100-500): {np.mean(vel_penalties):.4f}")
