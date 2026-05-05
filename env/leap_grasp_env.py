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
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32
        )

        # Object body id (needed to read pose)
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )

        # Lift threshold: object center must exceed this height to count as lifted
        self.lift_threshold = 0.25  # meters above world origin

        # Require cube to stay above threshold for this many consecutive steps
        # This prevents the "fling and terminate" exploit
        self.lift_sustain_required = 10
        self._lift_count = 0

        # Episode length
        self.max_steps = 500
        self._step_count = 0

        # Robustly find hand joint indices by name (avoids brittle slicing)
        self.hand_joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                                 for i in range(self.model.njnt)
                                 if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) != "object_joint"]
        
        self.hand_qpos_indices = np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.hand_joint_names])
            
        self.hand_qvel_indices = np.array([self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.hand_joint_names])

    # ------------------------------------------------------------------
    def _get_obs(self):
        joint_pos  = self.data.qpos[self.hand_qpos_indices].copy()
        joint_vel  = self.data.qvel[self.hand_qvel_indices].copy()
        obj_pos    = self.data.xpos[self.object_body_id] # (3,)  world position
        obj_quat   = self.data.xquat[self.object_body_id]# (4,)  world quaternion

        return np.concatenate([joint_pos, joint_vel, obj_pos, obj_quat]).astype(np.float32)

    # ------------------------------------------------------------------
    def _get_reward(self):
        obj_pos = self.data.xpos[self.object_body_id]

        # 1. Approach: positive reward for getting closer to the cube
        # mean_dist is ~0.14 open, ~0.10 closed.
        # 10.0 * (0.2 - mean_dist) -> +0.6 open, +1.0 closed.
        fingertip_names = ["if_tip", "mf_tip", "rf_tip", "th_tip"]
        distances = []
        for name in fingertip_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Required geom '{name}' not found in MuJoCo model.")
            tip_pos = self.data.geom_xpos[geom_id]
            distances.append(np.linalg.norm(tip_pos - obj_pos))
        mean_dist = np.mean(distances) if distances else 1.0
        approach_reward = 10.0 * max(0.0, 0.2 - mean_dist)

        # 2. Contact reward: detect actual hand-on-cube contacts
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        n_hand_contacts = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            is_cube_contact = (c.geom1 == cube_geom_id) or (c.geom2 == cube_geom_id)
            is_env_contact = (c.geom1 in [table_geom_id, floor_geom_id]) or (c.geom2 in [table_geom_id, floor_geom_id])
            
            if is_cube_contact and not is_env_contact:
                n_hand_contacts += 1
                
        contact_reward = 0.5 * min(n_hand_contacts, 4)  # cap at 4 contacts

        # 3. Lift: only counts when hand is physically grasping the object
        lift_height = max(0.0, obj_pos[2] - 0.075)
        grasping = n_hand_contacts >= 2
        lift_reward = lift_height * 50.0 * (1.0 if grasping else 0.0)

        # 4. Action penalty: small regularization to penalize large deltas (jitter)
        action_penalty = -0.01 * np.sum(np.square(getattr(self, 'last_action', np.zeros(16))))

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
        
        # Initialize control targets to current joint positions for delta control
        self.data.ctrl[:16] = self.data.qpos[:16].copy()
        self.last_action = np.zeros(16, dtype=np.float32)
        
        self._step_count = 0
        self._lift_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.last_action = action
        
        # Delta control: action in [-1, 1] scaled by max_delta (0.1 rad per step)
        max_delta = 0.1
        target_ctrl = self.data.ctrl[:16] + action * max_delta
        
        # Clip final target to physical joint limits
        target_ctrl = np.clip(target_ctrl, self.joint_limits_low, self.joint_limits_high)
        self.data.ctrl[:16] = target_ctrl

        # Advance simulation (5 physics steps per policy step)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs     = self._get_obs()
        reward  = self._get_reward()
        self._step_count += 1

        obj_pos = self.data.xpos[self.object_body_id]

        # Sustained lift: must stay above threshold for N consecutive steps
        # Prevents the "fling" exploit where cube briefly pops above the threshold
        if obj_pos[2] > self.lift_threshold:
            self._lift_count += 1
        else:
            self._lift_count = 0

        # Out of bounds termination (cube dropped or flung away)
        dist_xy = np.linalg.norm(obj_pos[:2])
        out_of_bounds = dist_xy > 0.1 or obj_pos[2] < 0.0
        
        timeout    = self._step_count >= self.max_steps
        terminated = self._lift_count >= self.lift_sustain_required
        
        if out_of_bounds:
            reward -= 50.0
            terminated = True
            
        truncated  = timeout and not terminated
        
        # Explicitly inject the true rigorous success flag
        info = {
            'is_success': self._lift_count >= self.lift_sustain_required
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="video_cam")
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()