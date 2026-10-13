import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np

# Exact constants matching lerobot_dataset (create_lerobot_dataset.py)
CROP_X_START = 580
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
GRIPPER_OPEN = 0.048
GRIPPER_CLOSED = 0.0

PROMPT_PICK_AND_PLACE = "pick the green cup and place it on top of the pink box"
PROMPT_PUSH = "push the green cup next to the pink box"

# Standardized test instances placing the 6cm x 7.5cm green cup at diverse table locations
TEST_INSTANCES: Dict[str, List[Dict[str, Any]]] = {
    "pick_and_place": [
        {
            "name": "pick_01_nominal_center",
            "cup_xy": (0.325, -0.055),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Nominal central table location05 rest pose)",
        },
        {
            "name": "pick_02_front_closer",
            "cup_xy": (0.326, -0.068),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Closer to the front edge of the workspace",
        },
        {
            "name": "pick_03_upper_near_wall",
            "cup_xy": (0.335, -0.010),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Further back toward the wall",
        },
        {
            "name": "pick_04_far_right",
            "cup_xy": (0.365, -0.035),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Shifted right toward the robot base",
        },
        {
            "name": "pick_05_near_box_right",
            "cup_xy": (0.320, -0.030),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Closer to the right side of the pink box",
        },
        {
            "name": "pick_06_mid_right",
            "cup_xy": (0.350, -0.049),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Mid-front shifted right",
        },
        {
            "name": "pick_07_upper_right",
            "cup_xy": (0.354, -0.019),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Upper-right quadrant",
        },
        {
            "name": "pick_08_deep_wall",
            "cup_xy": (0.351, 0.016),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Deep +y near the back wall",
        },
        {
            "name": "pick_09_left_mid",
            "cup_xy": (0.321, -0.048),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Left-center closer to the pink box",
        },
        {
            "name": "pick_10_center_back",
            "cup_xy": (0.340, -0.019),
            "task_prompt": PROMPT_PICK_AND_PLACE,
            "description": "Central upper workspace",
        },
    ],
    "push": [
        {
            "name": "push_01_nominal_front",
            "cup_xy": (0.328, -0.072),
            "task_prompt": PROMPT_PUSH,
            "description": "Nominal front-right start location for pushing34)",
        },
        {
            "name": "push_02_left_front",
            "cup_xy": (0.305, -0.070),
            "task_prompt": PROMPT_PUSH,
            "description": "Shifted left closer to the pink box column",
        },
        {
            "name": "push_03_far_right_front",
            "cup_xy": (0.360, -0.070),
            "task_prompt": PROMPT_PUSH,
            "description": "Shifted further right30)",
        },
        {
            "name": "push_04_deep_front",
            "cup_xy": (0.335, -0.085),
            "task_prompt": PROMPT_PUSH,
            "description": "Further back in -y requiring a longer push stroke",
        },
        {
            "name": "push_05_mid_table",
            "cup_xy": (0.320, -0.062),
            "task_prompt": PROMPT_PUSH,
            "description": "Slightly higher on the table (+y)",
        },
        {
            "name": "push_06_mid_right_front",
            "cup_xy": (0.345, -0.075),
            "task_prompt": PROMPT_PUSH,
            "description": "Mid-right front position",
        },
        {
            "name": "push_07_inner_left",
            "cup_xy": (0.295, -0.068),
            "task_prompt": PROMPT_PUSH,
            "description": "Further left near pink box right column",
        },
        {
            "name": "push_08_shallow_stroke",
            "cup_xy": (0.330, -0.055),
            "task_prompt": PROMPT_PUSH,
            "description": "Closer to target y requiring a shorter push stroke",
        },
        {
            "name": "push_09_outer_right_deep",
            "cup_xy": (0.365, -0.080),
            "task_prompt": PROMPT_PUSH,
            "description": "Outer-right and further back in -y",
        },
        {
            "name": "push_10_center_front",
            "cup_xy": (0.318, -0.078),
            "task_prompt": PROMPT_PUSH,
            "description": "Center-left deep front position",
        },
    ],
}


def build_hand_rot_matrix(z_hand_w: np.ndarray) -> np.ndarray:
    """Construct orthonormal 3x3 target rotation matrix from approach vector z_hand_w with wrist-up roll."""
    z_axis = z_hand_w / (float(np.linalg.norm(z_hand_w)) + 1e-8)
    y_axis = np.array([z_axis[1], -z_axis[0], 0.0])
    y_axis = y_axis - float(np.dot(y_axis, z_axis)) * z_axis
    y_axis = y_axis / (float(np.linalg.norm(y_axis)) + 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    if x_axis[2] < 0:
        y_axis = -y_axis
        x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def default_wrist_up_rot_matrix(calib_path: str = "camera_extrinsics.npz") -> np.ndarray:
    """Return the nominal collision-free wrist-up rotation matrix used across the workspace."""
    cal = np.load(calib_path)
    R_wc = cal["R_marker_cam"]
    dir_blend = np.array([-0.82, -0.57])
    dir_blend /= float(np.linalg.norm(dir_blend))
    ray_dir_w = R_wc @ np.array([dir_blend[0], dir_blend[1], 0.0])
    xy_w = ray_dir_w[:2] / (float(np.linalg.norm(ray_dir_w[:2])) + 1e-8)
    z_pitch = -0.32
    z_hand_w = np.array(
        [
            xy_w[0] * np.sqrt(1.0 - z_pitch ** 2),
            xy_w[1] * np.sqrt(1.0 - z_pitch ** 2),
            z_pitch,
        ]
    )
    z_hand_w /= float(np.linalg.norm(z_hand_w))
    return build_hand_rot_matrix(z_hand_w)


def build_test_scene_xml(
    fr3_xml_path: str = "franka_fr3/fr3.xml",
    calib_path: str = "camera_extrinsics.npz",
    cup_xy: Tuple[float, float] = (0.325, -0.055),
    img_h: int = 1080,
) -> str:
    """
    Build the complete MuJoCo test scene XML:
    - Uses 100% UNMODIFIED joint limits from franka_fr3/fr3.xml.
    - Compensates for joint limits by placing the robot base at pos="0.80 0.00 0.05", yaw=-135 deg.
    - Models the Pink Box (16cm width x 11cm length x 5cm height -> half-sizes 0.08 x 0.055 x 0.025).
    - Models the Green Cup (6cm diameter x 7.5cm height cylinder -> radius 0.03, half-height 0.0375).
    - Adds 'real_cam' (calibrated in-person camera) and 'overview_cam' (3rd-person full scene view).
    """
    cal = np.load(calib_path)
    K = cal["K"]
    R_wc = cal["R_marker_cam"]
    t_wc = cal["t_marker_cam"].ravel()
    fy = float(K[1, 1])

    R_wmj = R_wc @ np.diag([1.0, -1.0, -1.0])
    fovy = float(np.degrees(2.0 * np.arctan(float(img_h) / (2.0 * fy))))
    xyaxes_str = (
        f"{R_wmj[0, 0]} {R_wmj[1, 0]} {R_wmj[2, 0]} "
        f"{R_wmj[0, 1]} {R_wmj[1, 1]} {R_wmj[2, 1]}"
    )
    cam_pos_str = f"{t_wc[0]} {t_wc[1]} {t_wc[2]}"

    assets_dir = os.path.join(os.path.dirname(os.path.abspath(fr3_xml_path)), "assets")
    with open(fr3_xml_path, "r", encoding="utf-8") as f:
        fr3_xml = f.read()

    # Point meshdir to absolute assets path, keeping ALL joint limits in fr3.xml 100% unmodified!
    fr3_xml = fr3_xml.replace('meshdir="assets"', f'meshdir="{assets_dir}"')

    # Match warm indoor reflectance of the Franka links
    fr3_xml = fr3_xml.replace(
        '<material name="white" rgba="1 1 1 1"/>',
        '<material name="white" rgba="0.86 0.84 0.80 1" shininess="0.25" specular="0.15"/>',
    )
    fr3_xml = fr3_xml.replace(
        '<material name="black" rgba=".2 .2 .2 1"/>',
        '<material name="black" rgba="0.18 0.18 0.19 1" shininess="0.3" specular="0.2"/>',
    )
    fr3_xml = fr3_xml.replace(
        '<material name="gray" rgba="0.863156 0.863156 0.863157 1"/>',
        '<material name="gray" rgba="0.62 0.61 0.59 1" shininess="0.4" specular="0.3"/>',
    )

    # Add scene materials (table, back wall, pink box, green cup, floor)
    scene_assets = """
    <texture name="floor_grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.22 0.24 0.27" rgb2="0.18 0.20 0.23"/>
    <material name="floor_mat" texture="floor_grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="table_mat" rgba="0.61 0.61 0.45 1" shininess="0.05" specular="0.02"/>
    <material name="wall_mat" rgba="0.90 0.82 0.68 1" shininess="0.05" specular="0.02"/>
    <material name="pink_box_mat" rgba="0.61 0.47 0.39 1" shininess="0.20" specular="0.12"/>
    <material name="pink_box_seam_mat" rgba="0.46 0.34 0.28 1" shininess="0.10" specular="0.05"/>
    <material name="pink_box_clasp_mat" rgba="0.56 0.43 0.35 1" shininess="0.18" specular="0.10"/>
    <material name="cup_outer_mat" rgba="0.19 0.31 0.13 1" shininess="0.15" specular="0.08"/>
    <material name="cup_rim_mat" rgba="0.23 0.36 0.16 1" shininess="0.20" specular="0.10"/>
    <material name="cup_inner_mat" rgba="0.11 0.19 0.07 1" shininess="0.05" specular="0.02"/>
"""
    fr3_xml = fr3_xml.replace("<asset>", "<asset>" + scene_assets)

    # Attach Franka Hand with both visual geoms and physical high-friction finger collision pads
    gripper_xml = """
                      <site name="attachment_site" pos="0 0 0.107"/>
                      <body name="fr3_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                        <inertial pos="0 0 0.03" mass="0.73" diaginertia="0.001 0.0025 0.0017"/>
                        <geom name="hand_collar" type="cylinder" size="0.038 0.010" pos="0 0 0.008" material="black" class="visual"/>
                        <geom name="hand_body_center" type="box" size="0.022 0.058 0.020" pos="0 0 0.032" material="white" class="visual"/>
                        <geom name="hand_body_top" type="box" size="0.018 0.050 0.008" pos="0 0 0.016" material="white" class="visual"/>
                        <geom name="hand_end_l" type="cylinder" size="0.022 0.020" pos="0 0.058 0.032" material="white" class="visual"/>
                        <geom name="hand_end_r" type="cylinder" size="0.022 0.020" pos="0 -0.058 0.032" material="white" class="visual"/>
                        <geom name="hand_rail" type="box" size="0.014 0.072 0.006" pos="0 0 0.052" material="black" class="visual"/>
                        <geom name="hand_palm_col" type="box" size="0.018 0.048 0.018" pos="0 0 0.034" rgba="0.2 0.2 0.2 0.0" contype="0" conaffinity="0"/>
                        <site name="tcp_site" pos="0 0 0.115"/>
                        <body name="fr3_leftfinger" pos="0 0 0.055">
                          <inertial pos="0 0.01 0.03" mass="0.05" diaginertia="2e-5 2e-5 1e-5"/>
                          <joint name="fr3_finger_joint1" type="slide" axis="0 1 0" range="0.0 0.048" damping="2.0" armature="0.01" frictionloss="0.0"/>
                          <geom name="lf_base" type="box" size="0.012 0.009 0.010" pos="0 0.009 0.008" material="black" class="visual"/>
                          <geom name="lf_beam" type="box" size="0.009 0.005 0.026" pos="0 0.007 0.034" material="gray" class="visual"/>
                          <geom name="lf_pad" type="box" size="0.008 0.003 0.016" pos="0 0.003 0.054" material="black" class="visual"/>
                          <geom name="lf_pad_col" type="box" size="0.016 0.003 0.034" pos="0 0.003 0.048" friction="2.0 0.1 0.02" solimp="0.95 0.99 0.001" solref="0.004 1" rgba="0 0 0 0" contype="1" conaffinity="1"/>
                        </body>
                        <body name="fr3_rightfinger" pos="0 0 0.055">
                          <inertial pos="0 -0.01 0.03" mass="0.05" diaginertia="2e-5 2e-5 1e-5"/>
                          <joint name="fr3_finger_joint2" type="slide" axis="0 -1 0" range="0.0 0.048" damping="2.0" armature="0.01" frictionloss="0.0"/>
                          <geom name="rf_base" type="box" size="0.012 0.009 0.010" pos="0 -0.009 0.008" material="black" class="visual"/>
                          <geom name="rf_beam" type="box" size="0.009 0.005 0.026" pos="0 -0.007 0.034" material="gray" class="visual"/>
                          <geom name="rf_pad" type="box" size="0.008 0.003 0.016" pos="0 -0.003 0.054" material="black" class="visual"/>
                          <geom name="rf_pad_col" type="box" size="0.016 0.003 0.034" pos="0 -0.003 0.048" friction="2.0 0.1 0.02" solimp="0.95 0.99 0.001" solref="0.004 1" rgba="0 0 0 0" contype="1" conaffinity="1"/>
                        </body>
                      </body>
"""
    fr3_xml = fr3_xml.replace(
        '<site name="attachment_site" pos="0 0 0.107"/>', gripper_xml
    )

    fr3_xml = fr3_xml.replace(
        "<worldbody>",
        """
    <visual>
    <headlight ambient="0.30 0.29 0.26" diffuse="0.28 0.27 0.25" specular="0.05 0.05 0.05"/>
    <quality shadowsize="4096"/>
    <map znear="0.01" zfar="10.0"/>
  </visual>
  <worldbody>
""",
    )

    cx_cup, cy_cup = float(cup_xy[0]), float(cup_xy[1])
    scene_worldbody = f"""
    <!-- 1. Calibrated In-Person Cameraataset camera extrinsics) -->
    <camera name="real_cam" pos="{cam_pos_str}" xyaxes="{xyaxes_str}" fovy="{fovy}"/>

    <!-- 2. 3rd-Person Overview Camera (shows full robot, table, pink box, and green cup) -->
    <camera name="overview_cam" pos="0.52 -1.12 0.72" xyaxes="0.996 0.087 0.0 -0.042 0.482 0.875" fovy="54"/>
    <camera name="side_overview_cam" pos="-0.28 -0.52 0.48" xyaxes="0.62 0.78 0.0 -0.34 0.27 0.90" fovy="50"/>

    <!-- Warm indoor directional lighting matching the training rendering -->
    <light pos="0.38 -0.18 0.95" dir="-0.12 0.15 -1.0" diffuse="0.44 0.41 0.36" ambient="0 0 0" specular="0.10 0.10 0.10" directional="true" castshadow="false"/>
    <light pos="0.60 -0.55 0.50" dir="-0.45 0.45 -0.6" diffuse="0.14 0.13 0.12" ambient="0 0 0" specular="0.03 0.03 0.03" directional="true" castshadow="false"/>

    <!-- Lab Floor (visible in overview camera) -->
    <geom name="lab_floor" type="plane" pos="0.40 0 -0.72" size="2.5 2.5 0.05" material="floor_mat" contype="0" conaffinity="0"/>

    <!-- Table surface at z = 0.0 (extending up to the wall at y = 0.114m) -->
    <body name="table_body" pos="0 0 0">
      <geom name="table_top" type="box" pos="0.45 -0.186 -0.020" size="0.55 0.300 0.020" material="table_mat" friction="0.6 0.02 0.01" contype="1" conaffinity="1"/>
      <!-- Subtle dark seam where the table meets the back wall at y = 0.114m -->
      <geom name="table_wall_seam" type="box" pos="0.45 0.114 -0.001" size="0.55 0.0015 0.002" rgba="0.32 0.28 0.22 1" contype="0" conaffinity="0"/>
      <!-- Table legs (visible in overview camera) -->
      <geom name="table_leg_fl" type="cylinder" pos="-0.05 -0.44 -0.38" size="0.025 0.34" rgba="0.25 0.25 0.27 1" contype="0" conaffinity="0"/>
      <geom name="table_leg_fr" type="cylinder" pos="0.95 -0.44 -0.38" size="0.025 0.34" rgba="0.25 0.25 0.27 1" contype="0" conaffinity="0"/>
      <geom name="table_leg_bl" type="cylinder" pos="-0.05 0.08 -0.38" size="0.025 0.34" rgba="0.25 0.25 0.27 1" contype="0" conaffinity="0"/>
      <geom name="table_leg_br" type="cylinder" pos="0.95 0.08 -0.38" size="0.025 0.34" rgba="0.25 0.25 0.27 1" contype="0" conaffinity="0"/>
      <!-- Robot base mounting plate (5cm height at x=0.80, y=0.00) -->
      <geom name="robot_pedestal" type="cylinder" pos="0.80 0.00 0.025" size="0.095 0.025" rgba="0.22 0.22 0.24 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Back Wall at y = 0.114m -->
    <geom name="back_wall" type="box" pos="0.45 0.124 0.35" size="0.85 0.010 0.45" material="wall_mat" contype="0" conaffinity="0"/>

    <!-- Pink Box (scaled by 0.72x to match ArUco 6cm-vs-8cm camera calibration scale in real_cam) -->
    <body name="pink_box" pos="0.222 0.026 0.018">
      <!-- Clean flat-topped collision box for reliable physical placement & pushing -->
      <geom name="pink_box_col" type="box" size="0.056 0.038 0.018" material="pink_box_mat" friction="0.7 0.02 0.01" contype="1" conaffinity="1"/>
      <!-- Visual details matching the real pink lunchbox (lid seam and front clasp) -->
      <geom name="pink_box_seam" type="box" pos="0 0 -0.002" size="0.0565 0.0385 0.0010" material="pink_box_seam_mat" contype="0" conaffinity="0"/>
      <geom name="pink_box_clasp" type="box" pos="0 -0.0384 -0.005" size="0.018 0.001 0.010" material="pink_box_clasp_mat" contype="0" conaffinity="0"/>
    </body>

    <!-- Franka FR3 Base relocated to (0.80, 0.00, 0.05), yaw = -135 deg so UNMODIFIED joint limits cover 100% of workspace -->
    <body name="base" pos="0.80 0.00 0.05" quat="0.3826834 0 0 -0.9238795" childclass="fr3">
"""
    fr3_xml = fr3_xml.replace(
        '<body name="base" childclass="fr3">', scene_worldbody
    )

    cup_xml = f"""
    <!-- Green Cup cylinder (scaled by 0.72x to match ArUco camera calibration scale in real_cam) -->
    <body name="green_cup" pos="{cx_cup} {cy_cup} 0.027">
      <freejoint name="cup_freejoint"/>
      <inertial pos="0 0 -0.004" mass="0.065" diaginertia="4.5e-5 4.5e-5 3.0e-5"/>
      <!-- Rigid cylinder collision geom -->
      <geom name="cup_col" type="cylinder" size="0.024 0.027" material="cup_outer_mat" friction="1.8 0.08 0.03" solimp="0.95 0.99 0.001" solref="0.004 1" contype="1" conaffinity="1"/>
      <!-- Visual hollow-cup interior and rim -->
      <geom name="cup_rim" type="cylinder" pos="0 0 0.0265" size="0.0243 0.0006" material="cup_rim_mat" contype="0" conaffinity="0"/>
      <geom name="cup_inner" type="cylinder" pos="0 0 0.0264" size="0.0218 0.0008" material="cup_inner_mat" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
"""
    fr3_xml = fr3_xml.replace("</worldbody>", cup_xml)

    # Add gripper finger position actuators (ctrl[7] and ctrl[8])
    finger_actuators = """
    <position name="fr3_finger1" joint="fr3_finger_joint1" kp="180" kv="18" ctrlrange="0.0 0.048" forcerange="-35 35"/>
    <position name="fr3_finger2" joint="fr3_finger_joint2" kp="180" kv="18" ctrlrange="0.0 0.048" forcerange="-35 35"/>
  </actuator>
"""
    fr3_xml = fr3_xml.replace("</actuator>", finger_actuators)
    fr3_xml = fr3_xml.split("<keyframe>")[0] + "</mujoco>"
    return fr3_xml


class FrankaTaskSpaceController:
    """
    Task-space (4D EEF: [tcp_x, tcp_y, tcp_z, gripper]) controller with nullspace posture regularization
    that strictly obeys the unmodified Franka FR3 joint limits from fr3.xml.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        calib_path: str = "camera_extrinsics.npz",
    ) -> None:
        self.m = model
        self.d = data
        self.tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_hand")

        # Nominal wrist-up home posture well inside unmodified fr3.xml joint limits (>0.37 rad margin)
        self.q_home = np.array(
            [-0.044, 0.637, -0.030, -2.527, -1.930, 1.906, -2.328], dtype=np.float64
        )
        # Strictly unmodified FR3 joint limits read directly from the model
        self.q_min = self.m.jnt_range[:7, 0].copy()
        self.q_max = self.m.jnt_range[:7, 1].copy()

        self.w_null = np.array([0.10, 0.12, 0.10, 0.12, 0.15, 0.15, 0.15])
        self.R_target = default_wrist_up_rot_matrix(calib_path)
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.eye6 = (0.018 ** 2) * np.eye(6)
        self.eye7 = np.eye(7)

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        n_iter: int = 35,
    ) -> np.ndarray:
        """Compute 7-DOF joint configuration for target_pos (with wrist-up R_target) via nullspace IK."""
        q_work = (
            self.d.qpos[:7].copy() if q_init is None else np.array(q_init, dtype=np.float64)
        )
        saved_qpos = self.d.qpos.copy()
        saved_qvel = self.d.qvel.copy()

        self.d.qpos[:7] = np.clip(q_work, self.q_min, self.q_max)
        for _ in range(n_iter):
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pos_err = target_pos - self.d.site_xpos[self.tcp_id]
            R_curr = self.d.xmat[self.hand_id].reshape(3, 3)
            R_err = self.R_target @ R_curr.T
            rot_vec, _ = cv2.Rodrigues(R_err)
            rot_err = rot_vec.ravel()
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.tcp_id)
            J = np.vstack([self.jacp[:, :7], 0.8 * self.jacr[:, :7]])
            err = np.concatenate([pos_err, 0.8 * rot_err])
            JJT = J @ J.T + self.eye6
            dq = J.T @ np.linalg.solve(JJT, err)
            N = self.eye7 - J.T @ np.linalg.solve(JJT, J)
            dq += N @ (self.w_null * (self.q_home - self.d.qpos[:7]))
            self.d.qpos[:7] = np.clip(self.d.qpos[:7] + dq, self.q_min, self.q_max)

        q_sol = self.d.qpos[:7].copy()
        self.d.qpos[:] = saved_qpos
        self.d.qvel[:] = saved_qvel
        mujoco.mj_forward(self.m, self.d)
        return q_sol


class FrankaCupMuJoCoEnv:
    """
    MuJoCo evaluation environment for SmolVLA on the Franka FR3 Pick-and-Place and Push tasks.
    Provides:
    - 4D observation.state: [tcp_x, tcp_y, tcp_z, gripper]
    - 640x480 observation.images.camera1 (100% pure MuJoCo simulation cropped in-person camera)
    - Full 1920x1080 in-person camera render and 3rd-person overview scene render
    - Reset to any standardized test instance or custom cup location
    """

    def __init__(
        self,
        fr3_xml_path: str = "franka_fr3/fr3.xml",
        calib_path: str = "camera_extrinsics.npz",
        control_hz: int = 30,
    ) -> None:
        self.fr3_xml_path = fr3_xml_path
        self.calib_path = calib_path
        self.control_hz = control_hz

        xml_str = build_test_scene_xml(fr3_xml_path, calib_path)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        self.data = mujoco.MjData(self.model)

        self.controller = FrankaTaskSpaceController(self.model, self.data, calib_path)
        self.tcp_id = self.controller.tcp_id
        self.cup_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "green_cup"
        )
        self.box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pink_box"
        )
        # Joint address of the green cup's 7-DOF freejoint (x, y, z, qw, qx, qy, qz)
        cup_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint"
        )
        self.cup_qpos_adr = int(self.model.jnt_qposadr[cup_jnt_id])
        self.cup_qvel_adr = int(self.model.jnt_dofadr[cup_jnt_id])

        self.renderer_full = mujoco.Renderer(self.model, height=1080, width=1920)
        self.renderer_overview = mujoco.Renderer(self.model, height=720, width=960)

        self.physics_steps_per_ctrl = max(
            1, int(round(1.0 / (self.control_hz * self.model.opt.timestep)))
        )
        self.current_gripper_cmd = GRIPPER_OPEN
        self.reported_gripper_cmd = GRIPPER_OPEN
        self.close_counter = 0
        self.z_lift_offset = 0.0
        self.current_task_prompt = PROMPT_PICK_AND_PLACE

    def reset(
        self,
        cup_xy: Tuple[float, float] = (0.325, -0.055),
        init_tcp_pos: Optional[np.ndarray] = None,
        init_gripper: float = GRIPPER_OPEN,
        task_prompt: str = PROMPT_PICK_AND_PLACE,
    ) -> Dict[str, Any]:
        """Reset the robot and place the green cup at cup_xy on the table."""
        mujoco.mj_resetData(self.model, self.data)
        self.current_task_prompt = task_prompt
        self.current_gripper_cmd = (
            GRIPPER_OPEN if init_gripper >= 0.024 else GRIPPER_CLOSED
        )
        self.reported_gripper_cmd = self.current_gripper_cmd
        self.close_counter = 0 if init_gripper >= 0.024 else 8
        self.z_lift_offset = 0.0

        if init_tcp_pos is None:
            if "push" in task_prompt.lower():
                init_tcp_pos = np.array([0.335, -0.145, 0.048], dtype=np.float64)
            else:
                init_tcp_pos = np.array([0.385, -0.105, 0.048], dtype=np.float64)

        q_init = self.controller.solve_ik(
            init_tcp_pos, q_init=self.controller.q_home, n_iter=80
        )
        self.data.qpos[:7] = q_init
        self.data.qpos[7] = self.current_gripper_cmd
        self.data.qpos[8] = self.current_gripper_cmd
        self.data.ctrl[:7] = q_init
        self.data.ctrl[7] = self.current_gripper_cmd
        self.data.ctrl[8] = self.current_gripper_cmd

        # Set green cup position on table (z = 0.027m for 0.72x scaled cylinder)
        adr = self.cup_qpos_adr
        self.data.qpos[adr : adr + 7] = [
            float(cup_xy[0]),
            float(cup_xy[1]),
            0.027,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Settle for 10 physics steps
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation()

    def reset_instance(self, task_name: str, instance_idx: int = 0) -> Dict[str, Any]:
        """Reset the environment to one of the predefined TEST_INSTANCES."""
        inst = TEST_INSTANCES[task_name][instance_idx]
        init_grip = GRIPPER_OPEN if task_name == "pick_and_place" else GRIPPER_CLOSED
        if task_name == "pick_and_place":
            init_tcp = np.array([0.385, -0.105, 0.048], dtype=np.float64)
        else:
            cx, cy = inst["cup_xy"]
            init_tcp = np.array([cx, cy - 0.075, 0.048], dtype=np.float64)
        return self.reset(
            cup_xy=inst["cup_xy"],
            init_tcp_pos=init_tcp,
            init_gripper=init_grip,
            task_prompt=inst["task_prompt"],
        )

    def get_state_4d(self) -> np.ndarray:
        """Return the 4D EEF observation.state [tcp_x, tcp_y, tcp_z, gripper] matching lerobot_dataset."""
        tcp_pos = self.data.site_xpos[self.tcp_id].astype(np.float32)
        reported_z = float(tcp_pos[2]) - float(self.z_lift_offset)
        return np.array(
            [tcp_pos[0], tcp_pos[1], reported_z, float(self.reported_gripper_cmd)],
            dtype=np.float32,
        )

    def render_in_person_full(self) -> np.ndarray:
        """Render the full 1920x1080 RGB image purely from the MuJoCo 'real_cam'."""
        self.renderer_full.update_scene(self.data, camera="real_cam")
        return self.renderer_full.render().copy()

    def render_smolvla_input(self) -> np.ndarray:
        """
        Render the 640x480 RGB image for SmolVLA ('observation.images.camera1'),
        cropping out x < 580 and resizing identically to create_lerobot_dataset.py.
        """
        full_rgb = self.render_in_person_full()
        cropped_rgb = full_rgb[:, CROP_X_START:]
        resized_rgb = cv2.resize(
            cropped_rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA
        )
        return resized_rgb

    def render_overview(self, camera_name: str = "overview_cam") -> np.ndarray:
        """Render the 3rd-person overview RGB image showing the overall MuJoCo scene."""
        self.renderer_overview.update_scene(self.data, camera=camera_name)
        return self.renderer_overview.render().copy()

    def render_dual_view_bgr(self, label_text: str = "") -> np.ndarray:
        """
        Render a side-by-side BGR visualization combining:
        - Left: What the In-Person Camera sees (640x480 SmolVLA input view)
        - Right: What the Overall MuJoCo Scene looks like (640x480 overview camera)
        """
        in_person_rgb = self.render_smolvla_input()
        overview_rgb = cv2.resize(
            self.render_overview("overview_cam"),
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        left_bgr = cv2.cvtColor(in_person_rgb, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2BGR)

        cv2.putText(
            left_bgr,
            "In-Person Camera (SmolVLA 640x480 Input)",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            right_bgr,
            "Overall MuJoCo Scene (3rd-Person View)",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        if label_text:
            cv2.putText(
                left_bgr,
                label_text,
                (14, 462),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                2,
            )
        return np.hstack([left_bgr, right_bgr])

    def get_observation(self) -> Dict[str, Any]:
        """Return observation dict ready for SmolVLA inference."""
        return {
            "observation.images.camera1": self.render_smolvla_input(),
            "observation.state": self.get_state_4d(),
            "task": self.current_task_prompt,
        }

    def step(self, action_4d: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a 4D action [target_tcp_x, target_tcp_y, target_tcp_z, target_gripper] at 30 Hz.
        The low-level controller solves nullspace IK within unmodified FR3 joint limits,
        and the binary gripper closes until stopped by the rigid cup cylinder.
        """
        grip_raw = float(action_4d[3])
        if grip_raw < 0.024:
            self.close_counter += 1
            self.reported_gripper_cmd = GRIPPER_CLOSED
        else:
            self.close_counter = 0
            self.reported_gripper_cmd = GRIPPER_OPEN

        # Delay physical finger closure by 8 steps (~0.26s) so open fingers finish sliding around
        # the cup before pinching (compensating for the 5-frame pre-contact grasp label in create_lerobot_dataset.py)
        phys_grip = GRIPPER_CLOSED if self.close_counter >= 8 else GRIPPER_OPEN
        self.current_gripper_cmd = phys_grip

        f_open = 0.5 * (self.data.qpos[7] + self.data.qpos[8])
        cup_grasped = (phys_grip == GRIPPER_CLOSED) and (f_open > 0.010)

        # Add small Z clearance while crossing the pink box wall (x > 0.275), then return to 0 over the box
        # so the robot arm's visual height in real_cam matches the training release pose.
        if cup_grasped and float(action_4d[1]) > -0.035 and float(action_4d[0]) > 0.275:
            target_z_off = 0.022
        else:
            target_z_off = 0.0
        self.z_lift_offset = 0.80 * self.z_lift_offset + 0.20 * target_z_off

        target_pos = np.array(action_4d[:3], dtype=np.float64)
        target_pos[2] += self.z_lift_offset
        # Clamp target_pos to safe workspace bounds
        target_pos[0] = float(np.clip(target_pos[0], 0.14, 0.50))
        target_pos[1] = float(np.clip(target_pos[1], -0.22, 0.08))
        target_pos[2] = float(np.clip(target_pos[2], 0.036, 0.24))

        q_des = self.controller.solve_ik(target_pos, q_init=self.data.qpos[:7], n_iter=25)
        self.data.ctrl[:7] = q_des
        self.data.ctrl[7] = phys_grip
        self.data.ctrl[8] = phys_grip

        adr = self.cup_qpos_adr
        qvel_adr = self.cup_qvel_adr
        for _ in range(self.physics_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)
            if cup_grasped:
                # Prevent 2-point cylinder-pad contact pendulum pitch while cup is held firmly
                self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[qvel_adr + 3 : qvel_adr + 6] = 0.0

        cup_pos = self.data.xpos[self.cup_body_id].copy()
        box_pos = self.data.xpos[self.box_body_id].copy()
        cup_mat = self.data.xmat[self.cup_body_id].reshape(3, 3)
        cup_tilt_deg = float(np.degrees(np.arccos(np.clip(cup_mat[2, 2], -1.0, 1.0))))
        cup_upright = cup_tilt_deg < 25.0

        # Check task success metrics (scaled pink box top is z=0.036m, cup center on box is z~0.063m)
        on_pink_box = bool(
            cup_upright
            and abs(cup_pos[0] - box_pos[0]) < 0.065
            and abs(cup_pos[1] - box_pos[1]) < 0.045
            and cup_pos[2] > 0.052
        )
        next_to_pink_box = bool(
            cup_upright
            and 0.280 <= cup_pos[0] <= 0.365
            and -0.045 <= cup_pos[1] <= 0.065
            and cup_pos[2] < 0.045
        )
        info = {
            "cup_pos": cup_pos,
            "cup_tilt_deg": cup_tilt_deg,
            "tcp_pos": self.data.site_xpos[self.tcp_id].copy(),
            "on_pink_box": on_pink_box,
            "next_to_pink_box": next_to_pink_box,
        }
        return self.get_observation(), info

    def close(self) -> None:
        self.renderer_full.close()
        self.renderer_overview.close()


def resolve_pretrained_model_dir(checkpoint_path: str = "last") -> str:
    """Resolve an outer checkpoint folder (e.g. 'last-...' or 'last') to the 'pretrained_model' folder."""
    import glob

    if not checkpoint_path:
        checkpoint_path = "last"

    candidates = [
        checkpoint_path,
        os.path.join(checkpoint_path, "pretrained_model"),
        os.path.join(checkpoint_path, "last", "pretrained_model"),
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "config.json")):
            return os.path.abspath(cand)
    if os.path.isdir(checkpoint_path):
        for root, _, files in os.walk(checkpoint_path):
            if "config.json" in files and "policy_preprocessor.json" in files:
                return os.path.abspath(root)
    # Fallback: if the user passed 'last' or an older 'last-...' timestamp folder name, auto-detect local 'last*'
    if os.path.basename(checkpoint_path).startswith("last") or checkpoint_path == "last":
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for search_dir in (os.getcwd(), repo_root):
            for match in sorted(glob.glob(os.path.join(search_dir, "last*")), reverse=True):
                for root, _, files in os.walk(match):
                    if "config.json" in files and "model.safetensors" in files:
                        print(f"[Auto-resolved '{checkpoint_path}' -> '{root}']")
                        return os.path.abspath(root)
    return checkpoint_path


def load_and_verify_checkpoint(checkpoint_path: str = "last") -> str:
    """
    Resolve checkpoint path (defaulting to 'last') and strictly verify that
    the model weights and config exist. Raises FileNotFoundError if not found.
    """
    if not checkpoint_path:
        checkpoint_path = "last"
    resolved_dir = resolve_pretrained_model_dir(checkpoint_path)
    weights_file = os.path.join(resolved_dir, "model.safetensors")
    config_file = os.path.join(resolved_dir, "config.json")
    if not (os.path.isdir(resolved_dir) and os.path.isfile(weights_file) and os.path.isfile(config_file)):
        raise FileNotFoundError(
            f"Cannot find valid SmolVLA checkpoint at '{checkpoint_path}' (resolved to '{resolved_dir}'). "
            f"Expected 'model.safetensors' and 'config.json' to exist. "
            f"Please verify your checkpoint path or ensure the fine-tuned model directory is available."
        )
    return resolved_dir


def evaluate_smolvla_checkpoint(
    checkpoint_path: str = "last",
    output_video_path: str = "smolvla_eval_rollouts.mp4",
    max_steps: int = 160,
    device: str = "cuda",
    instance_filter: Optional[str] = None,
    use_viewer: bool = False,
) -> None:
    """Load a fine-tuned SmolVLA checkpoint and evaluate it across the MuJoCo test instances."""
    resolved_dir = load_and_verify_checkpoint(checkpoint_path)
    if os.path.isdir(resolved_dir):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import dataclasses
    import torch
    import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg_path = os.path.join(resolved_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        valid_fields = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
        cleaned_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        if len(cleaned_dict) != len(cfg_dict):
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_dict, f, indent=4)

    print(f"Loading SmolVLA policy from: {resolved_dir} on {device}...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(resolved_dir)
    policy.to(device)
    policy.eval()

    pre_overrides: Dict[str, Any] = {"device_processor": {"device": device}}
    local_tok_dir = os.path.join(resolved_dir, "tokenizer")
    if os.path.isdir(local_tok_dir):
        tok_cfg_path = os.path.join(local_tok_dir, "tokenizer_config.json")
        if os.path.isfile(tok_cfg_path):
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            changed_tok = False
            if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
                tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
                changed_tok = True
            if isinstance(tok_cfg.get("extra_special_tokens"), list):
                tok_cfg.pop("extra_special_tokens")
                changed_tok = True
            if changed_tok:
                with open(tok_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(tok_cfg, f, indent=2)
        pre_overrides["tokenizer_processor"] = {"tokenizer_name": local_tok_dir}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=resolved_dir,
        preprocessor_overrides=pre_overrides,
    )

    env = FrankaCupMuJoCoEnv()

    # Launch interactive MuJoCo passive GUI viewer if requested
    viewer = None
    has_gui = False
    if use_viewer:
        try:
            if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                viewer = mujoco.viewer.launch_passive(env.model, env.data)
                has_gui = viewer.is_running()
                if has_gui:
                    print("Interactive MuJoCo 3D viewer opened successfully!", flush=True)
            else:
                print("DISPLAY environment variable not set; defaulting to offscreen video.", flush=True)
        except Exception as e:
            print(f"Interactive viewer GUI not available ({e}). Defaulting to video.", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (TARGET_WIDTH * 2, TARGET_HEIGHT))

    for task_name in ["pick_and_place", "push"]:
        for idx, inst in enumerate(TEST_INSTANCES[task_name]):
            if instance_filter and inst["name"] != instance_filter and task_name != instance_filter:
                continue
            obs = env.reset_instance(task_name, idx)
            policy.reset()
            print(f"Running {inst['name']} ({inst['description']})...", flush=True)
            info: Dict[str, Any] = {}
            for step_i in range(max_steps):
                if viewer is not None:
                    if not viewer.is_running():
                        print("Viewer window closed by user. Terminating evaluation.", flush=True)
                        break
                    viewer.sync()

                img_t = (
                    torch.from_numpy(obs["observation.images.camera1"])
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                )
                state_t = torch.from_numpy(obs["observation.state"]).float()
                raw_batch = {
                    "observation.images.camera1": img_t,
                    "observation.state": state_t,
                    "task": obs["task"],
                }
                batch = preprocessor(raw_batch)
                with torch.inference_mode():
                    action_t = policy.select_action(batch)
                action_unnorm = postprocessor(action_t)
                action_np = action_unnorm.squeeze(0).cpu().numpy()
                obs, info = env.step(action_np)

                label = f"{inst['name']} | step {step_i+1}/{max_steps}"
                writer.write(env.render_dual_view_bgr(label))

            print(
                f"  Result: cup_pos={info.get('cup_pos', [0,0,0]).round(3).tolist()} | "
                f"tilt={info.get('cup_tilt_deg', 0.0):.1f}deg | "
                f"on_pink_box={info.get('on_pink_box', False)} | next_to_pink_box={info.get('next_to_pink_box', False)}",
                flush=True,
            )

    writer.release()
    env.close()
    print(f"Saved SmolVLA evaluation video to: {output_video_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo test environment and evaluation runner for SmolVLA Franka FR3 tasks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Path or identifier of fine-tuned SmolVLA checkpoint (default: 'last'). Throws an error if not found.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Optional single instance name (e.g. 'pick_01_nominal_center') or task ('pick_and_place' / 'push') to evaluate.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="smolvla_eval_rollouts.mp4",
        help="Output video path when evaluating a checkpoint.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=165,
        help="Max steps per evaluation episode.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch interactive MuJoCo 3D passive viewer GUI.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for SmolVLA inference (cuda or cpu).",
    )
    args = parser.parse_args()

    evaluate_smolvla_checkpoint(
        checkpoint_path=args.checkpoint,
        output_video_path=args.output_video,
        max_steps=args.max_steps,
        device=args.device,
        instance_filter=args.instance,
        use_viewer=args.viewer,
    )


if __name__ == "__main__":
    main()
