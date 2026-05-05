import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from env.leap_grasp_env import LeapGraspEnv

def make_env():
    env = LeapGraspEnv()
    return env

def main():
    os.makedirs("results", exist_ok=True)
    
    # Create evaluation environment
    env = DummyVecEnv([make_env])
    
    # Load normalization stats from training
    if os.path.exists("results/vecnormalize.pkl"):
        env = VecNormalize.load("results/vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
    
    # Load the best or final model
    model_path = "results/best_model/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "results/final_model.zip"
        
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    
    num_episodes = 100
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Running evaluation for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            done = dones[0]
            
            # If the episode ends before max steps, it means the lift condition was met
            # or it dropped the cube (which gets a huge penalty).
            # A success is when the reward is positive and it terminates early.
            
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # Determine success: 
        # 1. Early termination with positive reward > 500 means it successfully lifted the cube past the threshold!
        # 2. Reaching 500 steps with >1000 reward means it stably grasped it but didn't quite hit the Z threshold.
        if (ep_length < 500 and ep_reward > 500) or (ep_length == 500 and ep_reward > 1000):
            success_count += 1
            
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, Length = {ep_length}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = (success_count / num_episodes) * 100
    
    print("\n--- Evaluation Results ---")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_episodes})")
    
    # Save to file
    with open("results/eval_results.txt", "w") as f:
        f.write("--- Evaluation Results ---\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Deterministic: True\n\n")
        f.write(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
        f.write(f"Mean Episode Length: {mean_length:.2f}\n")
        f.write(f"Success Rate: {success_rate:.2f}% ({success_count}/{num_episodes})\n")
        
    print("Metrics saved to results/eval_results.txt")

if __name__ == "__main__":
    main()
