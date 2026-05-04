import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv

# Directories
os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/best_model", exist_ok=True)

def make_env():
    return Monitor(LeapGraspEnv(), filename="results/logs/monitor.csv")

def make_eval_env():
    return Monitor(LeapGraspEnv(), filename="results/logs/eval_monitor.csv")

# Vectorized + normalized environments
# VecNormalize normalizes rewards to ~N(0,1) so the value network can fit
env      = VecNormalize(DummyVecEnv([make_env]),      norm_obs=False, norm_reward=True, clip_reward=10.0)
eval_env = VecNormalize(DummyVecEnv([make_eval_env]), norm_obs=False, norm_reward=False, training=False)

# Callbacks
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path="results/checkpoints/",
    name_prefix="leap_ppo"
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

# PPO model — tuned for stability:
#   vf_coef 0.5→1.0  : value network gets more gradient signal
#   clip_range 0.2→0.15: less aggressive policy steps
#   ent_coef 0.01→0.005: slightly less entropy to let policy focus
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.005,
    vf_coef=1.0,
    verbose=1,
    tensorboard_log="results/tensorboard/"
)

# Verify before full run
print("Running 100-step verification...")
model.learn(total_timesteps=100)
print("Verification passed.")

# Full training
model.learn(
    total_timesteps=1_500_000,
    callback=[checkpoint_cb, eval_cb],
    reset_num_timesteps=False
)
model.save("results/final_model")
# Save the VecNormalize stats so they can be loaded at eval/deploy time
env.save("results/vecnormalize.pkl")