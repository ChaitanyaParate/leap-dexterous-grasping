import os
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import imageio

sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv

def main():
    # 1. Create the environment
    def make_env():
        # Increase steps for video if needed
        return LeapGraspEnv()
    
    venv = DummyVecEnv([make_env])
    
    # 2. Load the normalization stats
    stats_path = "results/vecnormalize.pkl"
    if os.path.exists(stats_path):
        venv = VecNormalize.load(stats_path, venv)
        venv.training = False
        venv.norm_reward = False

    # 3. Load the trained model
    model_path = "results/final_model.zip"
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=venv)

    # 4. Record episodes
    fps = 30
    video_path = "results/grasp_demonstration.mp4"
    writer = imageio.get_writer(video_path, fps=fps)
    
    print(f"Recording demonstration to {video_path}...")
    
    for ep in range(3): # Record 3 episodes
        obs = venv.reset()
        done = False
        
        # Access the underlying env for rendering
        env = venv.envs[0].unwrapped
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            
            # Capture frame
            frame = env.render()
            writer.append_data(frame)
            
        print(f"Episode {ep+1} recorded.")
    
    writer.close()
    print("Video recording complete.")

if __name__ == "__main__":
    main()
