import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.leap_grasp_env import LeapGraspEnv
import os

def main():
    env = DummyVecEnv([lambda: LeapGraspEnv()])
    stats_path = "results/vecnormalize.pkl"
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        
    model_path = "results/best_model/best_model.zip"
    if not os.path.exists(model_path):
        print("No best_model found yet.")
        return
        
    model = PPO.load(model_path, env=env)
    
    heights = []
    for _ in range(5):
        obs = env.reset()
        max_height = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # obj_pos is in the info or I can get it from the env
            # The env is vectorized, so I need to access the original env
            obj_pos = env.get_attr('mj_data')[0].qpos[0:3]
            height = obj_pos[2] - 0.05 # Height above table
            max_height = max(max_height, height)
            if done:
                break
        heights.append(max_height)
        print(f"Episode {len(heights)} Max Lift: {max_height*100:.2f} cm")

    print(f"\nAverage Max Lift: {np.mean(heights)*100:.2f} cm")

if __name__ == "__main__":
    main()
