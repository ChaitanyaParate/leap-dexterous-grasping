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
    # Create pure evaluation environment (no VecNormalize wrapper needed since norm_obs=False)
    env = DummyVecEnv([make_env])
    
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
        is_success = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            done = dones[0]
            
            # Check rigorous physical success condition from info dict
            if infos[0].get('is_success', False):
                is_success = True
            
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # Determine success rigorously
        if is_success:
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
