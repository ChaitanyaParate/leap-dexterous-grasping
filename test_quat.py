import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('mujoco_menagerie/leap_hand/grasp_scene.xml')
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'palm')
m.body_quat[bid] = np.array([1.0, 0, 0, 0])
m.body_pos[bid] = np.array([0, 0, 0.25])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("Quat: 1, 0, 0, 0 | Pos: 0, 0, 0.25")
for name in ['if_tip', 'mf_tip', 'rf_tip', 'th_tip']:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    print(f'{name}: {d.geom_xpos[gid].round(4)}')
print(f'palm: {d.xpos[bid].round(4)}')
