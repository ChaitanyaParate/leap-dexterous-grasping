import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os

class LeapGraspEnv(gym.Env):
    """
    Observation space (39 values):
      - joint positions:  16
      - joint velocities: 16
      - object position:   3
      - object quaternion: 4
    
    Action space (16 values):
      - target joint angles, one per actuator
      - clipped to each joint's limit
    """

    def __init__(self, render_mode=None):
        super().__init__()

        xml_path = os.path.join(
            os.path.dirname(__file__),
            "../mujoco_menagerie/leap_hand/grasp_scene.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.renderer = None

        # Joint limits for clipping actions
        self.joint_limits_low  = self.model.actuator_ctrlrange[:, 0]  # shape (16,)
        self.joint_limits_high = self.model.actuator_ctrlrange[:, 1]  # shape (16,)

        # Gymnasium spaces
        obs_low  = np.full(39, -np.inf, dtype=np.float32)
        obs_high = np.full(39,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=self.joint_limits_low.astype(np.float32),
            high=self.joint_limits_high.astype(np.float32),
            dtype=np.float32
        )

        # Object body id (needed to read pose)
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )

        # Lift threshold: object center must exceed this height to count as lifted
        self.lift_threshold = 0.25  # meters above world origin

        # Episode length
        self.max_steps = 500
        self._step_count = 0

    # ------------------------------------------------------------------
    def _get_obs(self):
        joint_pos  = self.data.qpos[:16].copy()          # 16 joint angles
        joint_vel  = self.data.qvel[:16].copy()          # 16 joint velocities
        obj_pos    = self.data.xpos[self.object_body_id] # (3,)  world position
        obj_quat   = self.data.xquat[self.object_body_id]# (4,)  world quaternion

        return np.concatenate([joint_pos, joint_vel, obj_pos, obj_quat]).astype(np.float32)

    # ------------------------------------------------------------------
    def _get_reward(self):
        obj_pos = self.data.xpos[self.object_body_id]

        # 1. Approach: mean distance from fingertip geoms to object
        fingertip_names = ["if_tip", "mf_tip", "rf_tip", "th_tip"]
        distances = []
        for name in fingertip_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if site_id >= 0:
                tip_pos = self.data.geom_xpos[site_id]
                distances.append(np.linalg.norm(tip_pos - obj_pos))
        mean_dist = np.mean(distances) if distances else 1.0
        approach_reward = -mean_dist  # w=1.0

        # 2. Contact bonus: reward each fingertip within 3 cm
        contact_reward = 0.1 * np.sum([d < 0.03 for d in distances])

        # 3. Lift: only counts when hand is near the object (grasping, not knocking)
        lift_height = max(0.0, obj_pos[2] - 0.075)
        grasping = mean_dist < 0.05
        lift_reward = lift_height * 20.0 * (1.0 if grasping else 0.0)

        # 4. Action penalty: penalize large commanded actions
        action_penalty = -0.001 * np.sum(np.square(self.data.ctrl[:16]))

        return approach_reward + contact_reward + lift_reward + action_penalty

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube position slightly
        rng = np.random.default_rng(seed)
        obj_x = rng.uniform(-0.02, 0.02)
        obj_y = 0.04 + rng.uniform(-0.02, 0.02)

        # Find the freejoint qpos index for the object
        # Freejoint stores: x y z qw qx qy qz  (7 values)
        obj_jnt_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        obj_qpos_idx = self.model.jnt_qposadr[obj_jnt_id]
        self.data.qpos[obj_qpos_idx:obj_qpos_idx+3] = [obj_x, obj_y, 0.075]
        self.data.qpos[obj_qpos_idx+3:obj_qpos_idx+7] = [1, 0, 0, 0]  # identity quat

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        # Clip action to joint limits and apply
        action = np.clip(action, self.joint_limits_low, self.joint_limits_high)
        self.data.ctrl[:] = action

        # Advance simulation (5 physics steps per policy step)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs     = self._get_obs()
        reward  = self._get_reward()
        self._step_count += 1

        obj_pos     = self.data.xpos[self.object_body_id]
        lifted      = obj_pos[2] > self.lift_threshold
        timeout     = self._step_count >= self.max_steps
        terminated  = lifted
        truncated   = timeout

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()