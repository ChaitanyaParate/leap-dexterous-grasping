import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv

def main():
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/best_model", exist_ok=True)

    def make_env():
        return Monitor(LeapGraspEnv(), filename="results/logs/monitor_resume.csv")

    def make_eval_env():
        return Monitor(LeapGraspEnv(), filename="results/logs/eval_monitor_resume.csv")

    dummy_env = DummyVecEnv([make_env])
    dummy_eval_env = DummyVecEnv([make_eval_env])

    # Load the normalization stats
    stats_path = "results/vecnormalize.pkl"
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        env = VecNormalize.load(stats_path, dummy_env)
        
        eval_env = VecNormalize.load(stats_path, dummy_eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("Error: vecnormalize.pkl not found.")
        sys.exit(1)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="results/checkpoints/",
        name_prefix="leap_ppo_resume"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="results/best_model/",
        log_path="results/logs/",
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    # Load the model
    model_path = "results/checkpoints/leap_ppo_resume_2677048_steps.zip"
        
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, tensorboard_log="results/tensorboard/")

    print("Resuming training for 1,000,000 timesteps...")
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=False,
        tb_log_name="PPO_resume"
    )

    model.save("results/final_model")
    env.save("results/vecnormalize.pkl")
    print("Resumed training finished.")

if __name__ == "__main__":
    main()
