import argparse
import os
import queue
import threading
import time
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import mujoco
import numpy as np


def build_fr3_xml(
    fr3_xml_path: str,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    fy: float,
    img_h: int = 1080,
) -> str:
    """Build the MuJoCo XML string for the Franka FR3 + Franka Hand with calibrated camera and lighting."""
    R_wmj = R_wc @ np.diag([1.0, -1.0, -1.0])
    fovy = float(np.degrees(2.0 * np.arctan(float(img_h) / (2.0 * fy))))
    xyaxes_str = (
        f"{R_wmj[0, 0]} {R_wmj[1, 0]} {R_wmj[2, 0]} "
        f"{R_wmj[0, 1]} {R_wmj[1, 1]} {R_wmj[2, 1]}"
    )
    cam_pos_str = f"{t_wc[0]} {t_wc[1]} {t_wc[2]}"

    assets_dir = os.path.join(os.path.dirname( os.path.abspath(fr3_xml_path)), "assets")
    with open(fr3_xml_path, "r", encoding="utf-8") as f:
        fr3_xml = f.read()

    fr3_xml = fr3_xml.replace('meshdir="assets"', f'meshdir="{assets_dir}"')
    # Extend forearm/wrist roll ranges so the wrist-up branch reaches all table positions smoothly
    fr3_xml = fr3_xml.replace('range="0.5445 4.5169"', 'range="0.5445 5.50"')
    fr3_xml = fr3_xml.replace('range="-2.8065 2.8065"', 'range="-3.80 3.80"')
    fr3_xml = fr3_xml.replace('range="-3.0159 3.0159"', 'range="-3.60 3.60"')

    # Match real indoor warm lighting and material reflectance (no harsh white clipping or shadow acne)
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

    gripper_xml = """
                      <site name="attachment_site" pos="0 0 0.107"/>
                      <body name="fr3_hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                        <geom name="hand_collar" type="cylinder" size="0.038 0.010" pos="0 0 0.008" material="black" class="visual"/>
                        <geom name="hand_body_center" type="box" size="0.022 0.058 0.020" pos="0 0 0.032" material="white" class="visual"/>
                        <geom name="hand_body_top" type="box" size="0.018 0.050 0.008" pos="0 0 0.016" material="white" class="visual"/>
                        <geom name="hand_end_l" type="cylinder" size="0.022 0.020" pos="0 0.058 0.032" material="white" class="visual"/>
                        <geom name="hand_end_r" type="cylinder" size="0.022 0.020" pos="0 -0.058 0.032" material="white" class="visual"/>
                        <geom name="hand_rail" type="box" size="0.014 0.072 0.006" pos="0 0 0.052" material="black" class="visual"/>
                        <site name="tcp_site" pos="0 0 0.115"/>
                        <body name="fr3_leftfinger" pos="0 0 0.055">
                          <joint name="fr3_finger_joint1" type="slide" axis="0 1 0" range="0.0 0.048"/>
                          <geom name="lf_base" type="box" size="0.012 0.009 0.010" pos="0 0.009 0.008" material="black" class="visual"/>
                          <geom name="lf_beam" type="box" size="0.009 0.005 0.026" pos="0 0.007 0.034" material="gray" class="visual"/>
                          <geom name="lf_pad" type="box" size="0.008 0.003 0.016" pos="0 0.003 0.054" material="black" class="visual"/>
                        </body>
                        <body name="fr3_rightfinger" pos="0 0 0.055">
                          <joint name="fr3_finger_joint2" type="slide" axis="0 -1 0" range="0.0 0.048"/>
                          <geom name="rf_base" type="box" size="0.012 0.009 0.010" pos="0 -0.009 0.008" material="black" class="visual"/>
                          <geom name="rf_beam" type="box" size="0.009 0.005 0.026" pos="0 -0.007 0.034" material="gray" class="visual"/>
                          <geom name="rf_pad" type="box" size="0.008 0.003 0.016" pos="0 -0.003 0.054" material="black" class="visual"/>
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
    <headlight ambient="0.28 0.27 0.25" diffuse="0.30 0.29 0.27" specular="0.06 0.06 0.06"/>
  </visual>
  <worldbody>
""",
    )
    fr3_xml = fr3_xml.replace(
        '<body name="base" childclass="fr3">',
        f"""
    <camera name="real_cam" pos="{cam_pos_str}" xyaxes="{xyaxes_str}" fovy="{fovy}"/>
    <light pos="0.25 -0.15 0.85" dir="-0.1 0.2 -1" diffuse="0.42 0.40 0.36" ambient="0 0 0" specular="0.12 0.12 0.12" directional="true" castshadow="false"/>
    <light pos="0.60 -0.50 0.40" dir="-0.5 0.5 -0.5" diffuse="0.15 0.14 0.13" ambient="0 0 0" specular="0.04 0.04 0.04" directional="true" castshadow="false"/>
    <body name="base" pos="0.82 -0.38 0.05" quat="0.3826834 0 0 0.9238795" childclass="fr3">
""",
    )
    fr3_xml = fr3_xml.split("<keyframe>")[0] + "</mujoco>"
    return fr3_xml


class FrankaIKSolver:
    """Damped least-squares 6-DOF IK solver with wrist-up nullspace posture regularization."""

    def __init__(self, model: Any, data: Any, tcp_id: int, hand_id: int) -> None:
        self.m = model
        self.d = data
        self.tcp_id = tcp_id
        self.hand_id = hand_id
        # Wrist-up / elbow-up nominal posture: q6 < pi keeps link6 8.2cm ABOVE link7
        self.q_home = np.array([-0.14, 0.70, 0.20, -1.95, 2.85, 2.05, -2.44])
        self.q_min = self.m.jnt_range[:7, 0].copy()
        self.q_max = self.m.jnt_range[:7, 1].copy()
        # Strictly restrict q6 < pi so the solver can never flip into the wrist-down branch
        self.q_max[5] = 2.85
        self.w_null = np.array([0.08, 0.12, 0.08, 0.12, 0.15, 0.15, 0.25])
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.eye6 = (0.02 ** 2) * np.eye(6)
        self.eye7 = np.eye(7)

    def _solve_single(
        self,
        target_pos: np.ndarray,
        target_mat: np.ndarray,
        q_init: np.ndarray,
        n_iter: int = 45,
    ) -> Tuple[np.ndarray, float]:
        self.d.qpos[:7] = q_init
        for _ in range(n_iter):
            mujoco.mj_forward(self.m, self.d)
            pos_err = target_pos - self.d.site_xpos[self.tcp_id]
            R_curr = self.d.xmat[self.hand_id].reshape(3, 3)
            R_err = target_mat @ R_curr.T
            rot_vec, _ = cv2.Rodrigues(R_err)
            rot_err = rot_vec.ravel()
            if np.linalg.norm(pos_err) < 1.5e-4 and np.linalg.norm(rot_err) < 1.5e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.tcp_id)
            J = np.vstack([self.jacp[:, :7], 0.8 * self.jacr[:, :7]])
            err = np.concatenate([pos_err, 0.8 * rot_err])
            JJT = J @ J.T + self.eye6
            dq = J.T @ np.linalg.solve(JJT, err)
            N = self.eye7 - J.T @ np.linalg.solve(JJT, J)
            dq += N @ (self.w_null * (self.q_home - self.d.qpos[:7]))
            self.d.qpos[:7] = np.clip(self.d.qpos[:7] + dq, self.q_min, self.q_max)
        mujoco.mj_forward(self.m, self.d)
        err_norm = float(np.linalg.norm(target_pos - self.d.site_xpos[self.tcp_id]))
        return self.d.qpos[:7].copy(), err_norm

    def solve(
        self,
        target_pos: np.ndarray,
        target_mat: np.ndarray,
        q_init: np.ndarray,
    ) -> np.ndarray:
        q_sol, err = self._solve_single(target_pos, target_mat, q_init, n_iter=45)
        if err > 0.002:
            q_sol2, err2 = self._solve_single(
                target_pos, target_mat, self.q_home, n_iter=80
            )
            if err2 < err:
                return q_sol2
        return q_sol


def extract_hand_pose_and_cup(
    orig: np.ndarray,
    clean: np.ndarray,
    mp_hands: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    kernel_close: np.ndarray,
    kernel_erode: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float, Optional[np.ndarray]]:
    """
    Detect human hand via MediaPipe and green cup via HSV segmentation.
    Returns (detected, tcp_pos_w, z_hand_w, q_finger, cup_mask_occ).
    """
    # Downscale to 640x360 for MediaPipe Hands (MediaPipe internally resizes to 256x256)
    small_bgr = cv2.resize(orig, (640, 360), interpolation=cv2.INTER_LINEAR)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(small_rgb)
    if not res.multi_hand_landmarks:
        return False, None, None, 0.0, None

    lm2d = res.multi_hand_landmarks[0].landmark
    pts2d = np.array([[lm2d[i].x * 1920.0, lm2d[i].y * 1080.0] for i in range(21)])

    # Check if the hand is barely entering/leaving the extreme bottom-right edge
    wrist_2d = pts2d[0]
    thumb_2d = 0.65 * pts2d[4] + 0.35 * pts2d[3]
    finger_2d = 0.50 * pts2d[8] + 0.25 * pts2d[7] + 0.25 * pts2d[12]
    tcp_2d = 0.5 * (thumb_2d + finger_2d)
    if tcp_2d[0] > 1890.0 or tcp_2d[1] > 1060.0:
        return False, None, None, 0.0, None

    # Segment the green cup in both orig and clean frames
    hsv_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    cup_mask_orig_raw = cv2.inRange(
        hsv_orig, np.array([34, 50, 35]), np.array([88, 255, 255])
    )
    cup_mask_orig = cv2.morphologyEx(
        cup_mask_orig_raw, cv2.MORPH_CLOSE, kernel_close
    )

    hsv_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    cup_mask_clean = cv2.inRange(
        hsv_clean, np.array([34, 48, 35]), np.array([88, 255, 255])
    )

    # Strict intersection eroded by 1px prevents both inpainting green halos and beige background leaks
    cup_mask_occ = cv2.erode(
        cv2.bitwise_and(cup_mask_orig_raw, cup_mask_clean),
        kernel_erode,
        iterations=1,
    )

    ys_c, xs_c = np.nonzero(cup_mask_orig)
    if len(xs_c) > 100:
        u_cup = float(np.median(xs_c))
        v_cup_bot = float(np.percentile(ys_c, 96))
        v_cup_mid = float(np.median(ys_c))
        z_support = 0.046 if (680.0 < u_cup < 1130.0 and v_cup_bot < 580.0) else 0.0
        ray_bot_cam = np.array([(u_cup - cx) / fx, (v_cup_bot - cy) / fy, 1.0])
        ray_bot_w = R_wc @ ray_bot_cam
        z_cup_cam = float((z_support - t_wc[2]) / ray_bot_w[2] - 0.032)
        dist_to_cup_2d = float(np.hypot(tcp_2d[0] - u_cup, tcp_2d[1] - v_cup_mid))
    else:
        z_cup_cam = 0.42
        dist_to_cup_2d = 999.0

    ray_tcp_cam = np.array([(tcp_2d[0] - cx) / fx, (tcp_2d[1] - cy) / fy, 1.0])
    ray_tcp_w = R_wc @ ray_tcp_cam
    min_z_table = (
        0.088 if (680.0 < tcp_2d[0] < 1130.0 and tcp_2d[1] < 580.0) else 0.048
    )
    z_max_table = float((min_z_table - t_wc[2]) / ray_tcp_w[2])

    z_tcp = (
        min(z_cup_cam, z_max_table)
        if dist_to_cup_2d < 180.0
        else min(0.42, z_max_table)
    )
    tcp_pos_cam = ray_tcp_cam * z_tcp
    tcp_pos_w = R_wc @ tcp_pos_cam + t_wc

    knuckle_2d = 0.5 * (pts2d[5] + pts2d[9])
    dir_2d = knuckle_2d - wrist_2d
    dir_2d /= float(np.linalg.norm(dir_2d)) + 1e-6
    dir_blend = 0.25 * dir_2d + 0.75 * np.array([-0.82, -0.57])
    dir_blend /= float(np.linalg.norm(dir_blend))

    # Pitch approach vector ~19 deg downward from above (z_pitch = -0.32 in world frame)
    # so the wrist and forearm stay high above the table plane
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

    pinch_px = float(np.linalg.norm(thumb_2d - finger_2d))
    width_m = (pinch_px / fx) * z_tcp
    if dist_to_cup_2d < 120.0 and width_m > 0.025:
        width_m = max(width_m, 0.076)
        # Advance grasp 2.0 cm along approach vector so fingers wrap around cup
        tcp_pos_w = tcp_pos_w + 0.020 * z_hand_w

    q_finger = float(np.clip(width_m * 0.5, 0.002, 0.046))
    return True, tcp_pos_w, z_hand_w, q_finger, cup_mask_occ


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


def process_video(
    orig_video_path: str,
    clean_video_path: str,
    output_video_path: str,
    calib_path: str = "camera_extrinsics.npz",
    fr3_xml_path: str = "franka_fr3/fr3.xml",
    min_active_frame: int = 0,
) -> None:
    """Overlay simulated MuJoCo Franka FR3 onto the clean inpainted video."""
    cal = np.load(calib_path)
    K = cal["K"]
    R_wc = cal["R_marker_cam"]
    t_wc = cal["t_marker_cam"].ravel()
    R_cw = cal["R_cam_marker"]
    t_cw = cal["t_cam_marker"].ravel()
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    cap_orig = cv2.VideoCapture(orig_video_path)
    cap_clean = cv2.VideoCapture(clean_video_path)
    if not cap_orig.isOpened() or not cap_clean.isOpened():
        raise RuntimeError(
            f"Could not open input videos: {orig_video_path}, {clean_video_path}"
        )

    fps = float(cap_clean.get(cv2.CAP_PROP_FPS))
    img_w = int(cap_clean.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap_clean.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_clean.get(cv2.CAP_PROP_FRAME_COUNT))

    fr3_xml = build_fr3_xml(fr3_xml_path, R_wc, t_wc, fy, img_h=img_h)
    model = mujoco.MjModel.from_xml_string(fr3_xml)
    model.vis.global_.offwidth = img_w
    model.vis.global_.offheight = img_h
    data = mujoco.MjData(model)

    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_hand")
    ik_solver = FrankaIKSolver(model, data, tcp_id, hand_id)
    renderer = mujoco.Renderer(model, height=img_h, width=img_w)

    mp_hands_mod: Any = getattr(mp, "solutions").hands
    mp_hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (img_w, img_h))
    write_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=32)

    def _writer_worker() -> None:
        while True:
            item = write_queue.get()
            if item is None:
                write_queue.task_done()
                break
            writer.write(item)
            write_queue.task_done()

    writer_thread = threading.Thread(target=_writer_worker, daemon=True)
    writer_thread.start()

    kernel_close = np.ones((7, 7), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)

    # Lookahead buffer of 5 frames to bridge 1-2 frame MediaPipe dropouts and smooth trajectories
    buf: List[
        Tuple[
            int,
            np.ndarray,
            bool,
            Optional[np.ndarray],
            Optional[np.ndarray],
            float,
            Optional[np.ndarray],
        ]
    ] = []

    q_curr = ik_solver.q_home.copy()
    smooth_pos: Optional[np.ndarray] = None
    smooth_z: Optional[np.ndarray] = None
    smooth_finger: Optional[float] = None

    t0 = time.time()
    f_idx = 0

    def _read_and_detect(idx: int) -> Optional[
        Tuple[
            int,
            np.ndarray,
            bool,
            Optional[np.ndarray],
            Optional[np.ndarray],
            float,
            Optional[np.ndarray],
        ]
    ]:
        ret1, orig_f = cap_orig.read()
        ret2, clean_f = cap_clean.read()
        if not ret1 or not ret2:
            return None
        if idx < min_active_frame:
            return (idx, clean_f, False, None, None, 0.0, None)
        det, pos_w, z_w, q_f, occ_mask = extract_hand_pose_and_cup(
            orig_f,
            clean_f,
            mp_hands,
            fx,
            fy,
            cx,
            cy,
            R_wc,
            t_wc,
            kernel_close,
            kernel_erode,
        )
        return (idx, clean_f, det, pos_w, z_w, q_f, occ_mask)

    # Pre-fill lookahead buffer
    for _ in range(5):
        item = _read_and_detect(f_idx)
        if item is None:
            break
        buf.append(item)
        f_idx += 1

    processed = 0
    while buf:
        # Bridge 1-frame or 2-frame dropout at buf[1] if buf[0] and buf[2] (or buf[3]) are detected
        if len(buf) >= 3 and buf[0][2] and not buf[1][2]:
            next_valid = None
            for k in range(2, len(buf)):
                if buf[k][2]:
                    next_valid = k
                    break
            if next_valid is not None:
                _, clean_1, _, _, _, _, _ = buf[1]
                _, _, _, pos_0, z_0, qf_0, occ_0 = buf[0]
                _, _, _, pos_k, z_k, qf_k, _ = buf[next_valid]
                if (
                    pos_0 is not None
                    and pos_k is not None
                    and z_0 is not None
                    and z_k is not None
                ):
                    alpha_i = 1.0 / float(next_valid)
                    pos_i = (1.0 - alpha_i) * pos_0 + alpha_i * pos_k
                    z_i = (1.0 - alpha_i) * z_0 + alpha_i * z_k
                    z_i /= float(np.linalg.norm(z_i)) + 1e-8
                    qf_i = (1.0 - alpha_i) * qf_0 + alpha_i * qf_k
                    buf[1] = (buf[1][0], clean_1, True, pos_i, z_i, qf_i, occ_0)

        curr_idx, clean_frame, det, pos_w, z_w, q_f, occ_mask = buf.pop(0)

        if not det or pos_w is None or z_w is None:
            # Hand not detected in scene -> no Franka robot rendered
            smooth_pos = None
            smooth_z = None
            smooth_finger = None
            q_curr = ik_solver.q_home.copy()
            write_queue.put(clean_frame)
        else:
            # Blend current detection with lookahead neighbors and EMA state for silky-smooth motion
            pos_acc = pos_w.copy()
            z_acc = z_w.copy()
            qf_acc = q_f
            w_sum = 1.0
            for k in range(min(2, len(buf))):
                if buf[k][2] and buf[k][3] is not None and buf[k][4] is not None:
                    wk = 0.45 if k == 0 else 0.20
                    pos_acc += wk * buf[k][3]
                    z_acc += wk * buf[k][4]
                    qf_acc += wk * buf[k][5]
                    w_sum += wk
            pos_avg = pos_acc / w_sum
            z_avg = z_acc / (float(np.linalg.norm(z_acc)) + 1e-8)
            qf_avg = qf_acc / w_sum

            if smooth_pos is None or smooth_z is None or smooth_finger is None:
                smooth_pos = pos_avg
                smooth_z = z_avg
                smooth_finger = qf_avg
            else:
                smooth_pos = 0.55 * pos_avg + 0.45 * smooth_pos
                smooth_z = 0.45 * z_avg + 0.55 * smooth_z
                smooth_z /= float(np.linalg.norm(smooth_z)) + 1e-8
                smooth_finger = 0.50 * qf_avg + 0.50 * smooth_finger

            R_target = build_hand_rot_matrix(smooth_z)
            q_curr = ik_solver.solve(smooth_pos, R_target, q_curr)
            data.qpos[7] = smooth_finger
            data.qpos[8] = smooth_finger
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera="real_cam")
            rgb_rob = renderer.render()
            renderer.enable_depth_rendering()
            depth_rob = renderer.render()
            renderer.disable_depth_rendering()

            rob_mask = depth_rob < 5.0
            if np.any(rob_mask):
                tcp_cam_actual = R_cw @ data.site_xpos[tcp_id] + t_cw
                z_tcp_actual = float(tcp_cam_actual[2])

                if occ_mask is not None:
                    cup_occludes = (occ_mask > 0) & (depth_rob > z_tcp_actual)
                    visible_rob = rob_mask & (~cup_occludes)
                else:
                    visible_rob = rob_mask

                ys_r, xs_r = np.nonzero(visible_rob)
                if len(xs_r) > 0:
                    y0 = max(0, int(ys_r.min()) - 3)
                    y1 = min(img_h, int(ys_r.max()) + 4)
                    x0 = max(0, int(xs_r.min()) - 3)
                    x1 = min(img_w, int(xs_r.max()) + 4)

                    comp = clean_frame.copy()
                    sub_vis = visible_rob[y0:y1, x0:x1].astype(np.float32)
                    alpha = cv2.GaussianBlur(sub_vis, (3, 3), 0)[..., None]
                    sub_rob_bgr = cv2.cvtColor(
                        rgb_rob[y0:y1, x0:x1], cv2.COLOR_RGB2BGR
                    )
                    sub_clean = clean_frame[y0:y1, x0:x1]
                    comp[y0:y1, x0:x1] = (
                        alpha * sub_rob_bgr + (1.0 - alpha) * sub_clean
                    ).astype(np.uint8)
                    write_queue.put(comp)
                else:
                    write_queue.put(clean_frame)
            else:
                write_queue.put(clean_frame)

        next_item = _read_and_detect(f_idx)
        if next_item is not None:
            buf.append(next_item)
            f_idx += 1

        processed += 1
        if processed % 500 == 0 or processed == total_frames:
            elapsed = time.time() - t0
            fps_proc = processed / max(elapsed, 1e-6)
            eta = (total_frames - processed) / max(fps_proc, 1e-6)
            print(
                f"[{os.path.basename(output_video_path)}] "
                f"Frame {processed}/{total_frames} ({fps_proc:.1f} fps, ETA {eta:.1f}s)",
                flush=True,
            )

    write_queue.put(None)
    writer_thread.join()
    writer.release()
    cap_orig.release()
    cap_clean.release()
    mp_hands.close()
    renderer.close()
    print(f"Saved overlaid video to: {output_video_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay simulated Franka FR3 in MuJoCo onto masked videos."
    )
    parser.add_argument(
        "--task",
        choices=["all", "pick_and_place", "push"],
        default="all",
        help="Which video task to process.",
    )
    args = parser.parse_args()

    tasks = []
    if args.task in ("all", "pick_and_place"):
        tasks.append(
            (
                "raw_pick_and_place.mp4",
                "masked_clean_pick_and_place.mp4",
                "franka_overlay_pick_and_place.mp4",
                140,
            )
        )
    if args.task in ("all", "push"):
        tasks.append(
            (
                "raw_push.mp4",
                "masked_clean_push.mp4",
                "franka_overlay_push.mp4",
                0,
            )
        )

    for orig_p, clean_p, out_p, min_f in tasks:
        process_video(
            orig_video_path=orig_p,
            clean_video_path=clean_p,
            output_video_path=out_p,
            min_active_frame=min_f,
        )


if __name__ == "__main__":
    main()

