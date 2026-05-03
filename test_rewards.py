import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.insert(0, '.')
from env.leap_grasp_env import LeapGraspEnv
import mujoco

def make_env():
    return LeapGraspEnv()

venv = DummyVecEnv([make_env])
venv = VecNormalize.load('results/vecnormalize.pkl', venv)
venv.training = False
venv.norm_reward = False
model = PPO.load('results/final_model.zip', env=venv)

obs = venv.reset()
raw_env = venv.envs[0].unwrapped

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    
    obj_pos = raw_env.data.xpos[raw_env.object_body_id]
    
    fingertip_names = ["if_tip", "mf_tip", "rf_tip", "th_tip"]
    distances = []
    for name in fingertip_names:
        geom_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        tip_pos = raw_env.data.geom_xpos[geom_id]
        distances.append(np.linalg.norm(tip_pos - obj_pos))
    mean_dist = np.mean(distances)
    approach_reward = 10.0 * max(0.0, 0.2 - mean_dist)
    
    cube_geom_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    n_cube_contacts = 0
    for i in range(raw_env.data.ncon):
        c = raw_env.data.contact[i]
        if c.geom1 == cube_geom_id or c.geom2 == cube_geom_id:
            n_cube_contacts += 1
    contact_reward = 0.5 * min(n_cube_contacts, 4)
    
    lift_height = max(0.0, obj_pos[2] - 0.075)
    grasping = mean_dist < 0.06
    lift_reward = lift_height * 50.0 * (1.0 if grasping else 0.0)
    
    action_penalty = -0.01 * np.sum(np.square(raw_env.last_action))
    
    if step % 50 == 0:
        print(f"Step {step}: Total={reward[0]:.2f} | Approach={approach_reward:.2f} | Contact={contact_reward:.2f} ({n_cube_contacts} contacts) | Lift={lift_reward:.2f} | z={obj_pos[2]:.3f} | dist={mean_dist:.3f}")
