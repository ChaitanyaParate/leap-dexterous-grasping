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
        # Palm is at Z=0.16m, cube starts at Z=0.075m. 0.085m is just 1cm above spawn.
        self.lift_threshold = 0.085  # meters above world origin

        # Require cube to stay above threshold for this many consecutive steps
        # This prevents the "fling and terminate" exploit
        self.lift_sustain_required = 25
        self._lift_count = 0

        # Episode length
        self.max_steps = 1000
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
    @staticmethod
    def _rotate_vec(q, v):
        """Rotate vector v by quaternion q in MuJoCo [w, x, y, z] format."""
        w, x, y, z = q
        q_vec = np.array([x, y, z])
        return v + 2 * w * np.cross(q_vec, v) + 2 * np.cross(q_vec, np.cross(q_vec, v))

    # ------------------------------------------------------------------
    def _get_reward(self):
        obj_pos = self.data.xpos[self.object_body_id]

        # Approach reward removed: agent is fully trained and approach reward
        # creates lateral finger forces that push the cube out of reach.
        approach_reward = 0.0

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

        # 3. Lift: ZERO reward below threshold — forces agent to actually cross the line.
        # +10 per step when held above threshold while grasping.
        grasping = n_hand_contacts >= 2
        if obj_pos[2] > self.lift_threshold and grasping:
            lift_reward = 10.0
        else:
            lift_reward = 0.0

        # 4. Cube corner airborne reward: ensure all 8 cube corners stay above table.
        # Cube half-size = 0.02m. Table surface is at Z ~= 0.055m.
        # Reward proportional to how high the LOWEST corner is above the table.
        CUBE_HALF = 0.020
        TABLE_Z   = 0.055
        obj_quat  = self.data.xquat[self.object_body_id]  # [w, x, y, z]
        corner_signs = np.array([[sx, sy, sz]
                                  for sx in (-1, 1)
                                  for sy in (-1, 1)
                                  for sz in (-1, 1)], dtype=np.float64)
        world_corners = np.array([
            obj_pos + self._rotate_vec(obj_quat, CUBE_HALF * s)
            for s in corner_signs
        ])
        min_corner_z = float(np.min(world_corners[:, 2]))
        airborne_reward = 3.0 * max(0.0, min_corner_z - TABLE_Z)

        # Cube XY-drift penalty: strongly penalize if cube is pushed far from spawn
        spawn_xy = getattr(self, '_spawn_xy', np.zeros(2))
        xy_drift = np.linalg.norm(obj_pos[:2] - spawn_xy)
        drift_penalty = -15.0 * max(0.0, xy_drift - 0.03)

        # Spawn-height tether: penalize if cube falls below its spawn Z (0.085m).
        # Prevents the cube being dragged DOWN to the table by gravity during grasping.
        spawn_z = 0.085
        drop_below_spawn = max(0.0, spawn_z - obj_pos[2])
        spawn_z_penalty = -20.0 * drop_below_spawn

        # 5. Time penalty: -0.5 per step so the agent cannot profit from doing nothing.
        time_penalty = -0.5

        # 6. Action penalty: small regularization to penalize large deltas (jitter)
        action_penalty = -0.01 * np.sum(np.square(getattr(self, 'last_action', np.zeros(16))))

        return (contact_reward + lift_reward + airborne_reward
                + drift_penalty + spawn_z_penalty + time_penalty + action_penalty)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube position slightly — keep tight so cube stays under palm
        rng = np.random.default_rng(seed)
        obj_x = rng.uniform(-0.01, 0.01)
        obj_y = 0.04 + rng.uniform(-0.01, 0.01)

        # Find the freejoint qpos index for the object
        # Freejoint stores: x y z qw qx qy qz  (7 values)
        obj_jnt_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        obj_qpos_idx = self.model.jnt_qposadr[obj_jnt_id]
        self.data.qpos[obj_qpos_idx:obj_qpos_idx+3] = [obj_x, obj_y, 0.085]
        self.data.qpos[obj_qpos_idx+3:obj_qpos_idx+7] = [1, 0, 0, 0]  # identity quat

        mujoco.mj_forward(self.model, self.data)
        
        # Store spawn XY to detect drift
        self._spawn_xy = np.array([obj_x, obj_y])
        
        # Initialize control targets to current joint positions for delta control
        self.data.ctrl[:16] = self.data.qpos[:16].copy()
        self.last_action = np.zeros(16, dtype=np.float32)
        
        self._step_count = 0
        self._lift_count = 0
        self._has_succeeded = False
        self._was_above_threshold = False  # tracks if cube was lifted last step
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.last_action = action
        
        # Delta control: action in [-1, 1] scaled by max_delta (0.1 rad per step)
        max_delta = 0.05  # Reduced for smoother, more stable finger control
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
        currently_above = obj_pos[2] > self.lift_threshold
        if currently_above:
            self._lift_count += 1
        else:
            self._lift_count = 0

        # Drop penalty: punish losing the grip after having achieved the lift
        grasping_now = reward > 0  # proxy — will be overridden below properly
        if self._was_above_threshold and not currently_above:
            reward -= 20.0  # heavy penalty for dropping the cube
        self._was_above_threshold = currently_above

        # Out of bounds termination (cube dropped or flung away)
        dist_xy = np.linalg.norm(obj_pos[:2])
        out_of_bounds = dist_xy > 0.20 or obj_pos[2] < 0.0

        if self._lift_count >= self.lift_sustain_required:
            self._has_succeeded = True

        timeout    = self._step_count >= self.max_steps
        terminated = False
        
        if out_of_bounds:
            reward -= 50.0
            terminated = True
            
        truncated  = timeout and not terminated

        # Explicitly inject the true rigorous success flag
        info = {
            'is_success': getattr(self, '_has_succeeded', False)
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