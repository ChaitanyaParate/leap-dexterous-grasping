import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('mujoco_menagerie/leap_hand/grasp_scene.xml')
d = mujoco.MjData(m)

mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)

# Close fingers slowly
print("Closing fingers slowly:")
d.ctrl[:] = m.actuator_ctrlrange[:, 1]
max_vel_sq = 0
for _ in range(500):
    mujoco.mj_step(m, d)
    vel_sq = np.sum(np.square(d.qvel[:16]))
    max_vel_sq = max(max_vel_sq, vel_sq)
print(f"Max velocity squared: {max_vel_sq:.2f}")
print(f"Max velocity penalty (at 0.0005 coeff): {-0.0005 * max_vel_sq:.4f}")

# Fling cube
print("\nFlinging cube:")
mujoco.mj_resetData(m, d)
d.ctrl[:] = m.actuator_ctrlrange[:, 1] # snap shut instantly
max_vel_sq = 0
for i in range(500):
    if i == 10:
        d.ctrl[:] = m.actuator_ctrlrange[:, 0] # snap open
    mujoco.mj_step(m, d)
    vel_sq = np.sum(np.square(d.qvel[:16]))
    max_vel_sq = max(max_vel_sq, vel_sq)
print(f"Max velocity squared: {max_vel_sq:.2f}")
print(f"Max velocity penalty (at 0.0005 coeff): {-0.0005 * max_vel_sq:.4f}")
