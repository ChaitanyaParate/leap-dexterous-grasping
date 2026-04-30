# run this as: python3 -c "..." or save as test_env.py
import sys
sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv
import numpy as np

env = LeapGraspEnv()
obs, _ = env.reset()
print("Obs shape:", obs.shape)       # must be (39,)
print("Action space:", env.action_space.shape)  # must be (16,)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step {i}: reward={reward:.4f}, obj_z={obs[34]:.4f}")

env.close()
print("Environment test passed.")