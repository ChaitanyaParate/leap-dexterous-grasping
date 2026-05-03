import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys

sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv

def main():
    # 1. Create the environment
    # We use a DummyVecEnv because VecNormalize requires it
    def make_env():
        return LeapGraspEnv()
    
    venv = DummyVecEnv([make_env])
    
    # 2. Load the normalization stats
    stats_path = "results/vecnormalize.pkl"
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        venv = VecNormalize.load(stats_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("Warning: vecnormalize.pkl not found. Results may be degraded.")

    # 3. Load the trained model
    model_path = "results/final_model.zip"
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=venv)

    # 4. Run visualization loop with MuJoCo Viewer
    print("Launching MuJoCo Viewer...")
    
    # Get the unwrapped model/data from the first env
    raw_env = venv.envs[0].unwrapped
    
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        print("Viewer launched. Starting simulation...")
        
        while viewer.is_running():
            obs = venv.reset()
            done = False
            step_start = time.time()
            
            while not done and viewer.is_running():
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, done, info = venv.step(action)
                
                # Sync viewer
                viewer.sync()
                
                # Real-time synchronization
                time_until_next_step = raw_env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                step_start = time.time()
                
            print("Episode finished, resetting...")

if __name__ == "__main__":
    main()
