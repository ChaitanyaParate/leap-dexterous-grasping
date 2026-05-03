import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv

def make_env():
    return LeapGraspEnv()

venv = DummyVecEnv([make_env])
venv = VecNormalize.load('results/vecnormalize.pkl', venv)
venv.training = False
venv.norm_reward = False
model = PPO.load('results/final_model.zip', env=venv)

obs = venv.reset()
raw_env = venv.envs[0].unwrapped

print("First 10 actions:")
for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    print(f"Step {step}: mean={np.mean(action):.3f}, std={np.std(action):.3f}, min={np.min(action):.3f}, max={np.max(action):.3f}")
    if step > 0:
        delta = action - last_action
        print(f"  Max delta from previous step: {np.max(np.abs(delta)):.3f}")
    last_action = action
