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

ep_reward = 0
for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    ep_reward += reward[0]
    
    obj_pos = raw_env.data.xpos[raw_env.object_body_id]
    obj_z = obj_pos[2]
    dist = np.linalg.norm(obj_pos[:2]) # distance from origin (x,y)
    
    if step % 50 == 0 or done:
        print(f"Step {step}: obj_z={obj_z:.3f}, dist_xy={dist:.3f}, reward={reward[0]:.2f}")
        
    if done:
        print(f"Episode finished at step {step} with total reward {ep_reward:.2f}")
        break
