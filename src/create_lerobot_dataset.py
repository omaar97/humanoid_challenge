import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.ndimage import median_filter

from overlay_franka_mujoco import (
    FrankaIKSolver,
    build_fr3_xml,
    build_hand_rot_matrix,
)

# Binary gripper states (in meters of MuJoCo fr3_finger_joint1 slide range [0.0, 0.048])
GRIPPER_OPEN = 0.048
GRIPPER_CLOSED = 0.0

# Crop left side (x < 580) to completely remove the ArUco marker sheet, then resize to 640x480
CROP_X_START = 580
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

PROMPT_PICK_AND_PLACE = "pick the green cup and place it on top of the pink box"
PROMPT_PUSH = "push the green cup next to the pink box"


def extract_or_load_raw_trajectory(
    orig_video_path: str,
    calib_path: str,
    min_active_frame: int,
    cache_npz_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Extract per-frame hand and cup detections or load from cache if available."""
    if cache_npz_path is not None and os.path.exists(cache_npz_path):
        data = np.load(cache_npz_path)
        return {k: data[k] for k in data.files}

    cal = np.load(calib_path)
    K = cal["K"]
    R_wc = cal["R_marker_cam"]
    t_wc = cal["t_marker_cam"].ravel()
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    cap = cv2.VideoCapture(orig_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_hands_mod: Any = getattr(mp, "solutions").hands
    mp_hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    det_arr = np.zeros(total, dtype=bool)
    tcp_2d_arr = np.zeros((total, 2), dtype=np.float32)
    cup_2d_arr = np.zeros((total, 3), dtype=np.float32)
    tcp_w_arr = np.zeros((total, 3), dtype=np.float32)
    z_w_arr = np.zeros((total, 3), dtype=np.float32)
    qf_arr = np.zeros(total, dtype=np.float32)
    pinch_arr = np.zeros(total, dtype=np.float32)

    for idx in range(total):
        ret, orig = cap.read()
        if not ret:
            break
        small = cv2.resize(orig, (640, 360), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cmask = cv2.inRange(hsv, np.array([34, 50, 35]), np.array([88, 255, 255]))
        ys_c, xs_c = np.nonzero(cmask)
        if len(xs_c) > 12:
            u_cup = float(np.median(xs_c)) * 3.0
            v_cup_bot = float(np.percentile(ys_c, 96)) * 3.0
            v_cup_mid = float(np.median(ys_c)) * 3.0
            cup_2d_arr[idx] = [u_cup, v_cup_mid, v_cup_bot]

        if idx < min_active_frame:
            continue

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue

        lm2d = res.multi_hand_landmarks[0].landmark
        pts2d = np.array([[lm2d[i].x * 1920.0, lm2d[i].y * 1080.0] for i in range(21)])
        wrist_2d = pts2d[0]
        thumb_2d = 0.65 * pts2d[4] + 0.35 * pts2d[3]
        finger_2d = 0.50 * pts2d[8] + 0.25 * pts2d[7] + 0.25 * pts2d[12]
        tcp_2d = 0.5 * (thumb_2d + finger_2d)
        if tcp_2d[0] > 1890.0 or tcp_2d[1] > 1060.0:
            continue

        det_arr[idx] = True
        tcp_2d_arr[idx] = tcp_2d
        pinch_px = float(np.linalg.norm(thumb_2d - finger_2d))
        pinch_arr[idx] = pinch_px

        if cup_2d_arr[idx, 0] > 0:
            u_cup, v_cup_mid, v_cup_bot = cup_2d_arr[idx]
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

        width_m = (pinch_px / fx) * z_tcp
        if dist_to_cup_2d < 120.0 and width_m > 0.025:
            width_m = max(width_m, 0.076)
            tcp_pos_w = tcp_pos_w + 0.020 * z_hand_w
        q_finger = float(np.clip(width_m * 0.5, 0.002, 0.046))

        tcp_w_arr[idx] = tcp_pos_w
        z_w_arr[idx] = z_hand_w
        qf_arr[idx] = q_finger

    cap.release()
    mp_hands.close()

    out = {
        "det": det_arr,
        "tcp_2d": tcp_2d_arr,
        "cup_2d": cup_2d_arr,
        "tcp_w": tcp_w_arr,
        "z_w": z_w_arr,
        "qf": qf_arr,
        "pinch": pinch_arr,
    }
    if cache_npz_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_npz_path)), exist_ok=True)
        np.savez(cache_npz_path, **out)
    return out


def compute_full_joint_trajectory(
    traj: Dict[str, np.ndarray],
    ik_solver: FrankaIKSolver,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the exact smoothed TCP and 7-DOF joint trajectory from overlay_franka_mujoco.py.
    Returns (det_bridged, tcp_2d_bridged, q_joints_all, smooth_tcp_all).
    """
    det = traj["det"].copy()
    tcp_2d = traj["tcp_2d"].copy()
    tcp_w = traj["tcp_w"].copy()
    z_w = traj["z_w"].copy()
    N = len(det)

    # Bridge 1- or 2-frame dropouts identically to overlay_franka_mujoco.py
    for idx in range(N - 3):
        if det[idx] and not det[idx + 1]:
            next_valid = None
            for k in range(idx + 2, min(N, idx + 4)):
                if det[k]:
                    next_valid = k
                    break
            if next_valid is not None:
                alpha = 1.0 / float(next_valid - idx)
                tcp_2d[idx + 1] = (1.0 - alpha) * tcp_2d[idx] + alpha * tcp_2d[next_valid]
                tcp_w[idx + 1] = (1.0 - alpha) * tcp_w[idx] + alpha * tcp_w[next_valid]
                z_interp = (1.0 - alpha) * z_w[idx] + alpha * z_w[next_valid]
                z_w[idx + 1] = z_interp / (float(np.linalg.norm(z_interp)) + 1e-8)
                det[idx + 1] = True

    q_joints_all = np.zeros((N, 7), dtype=np.float32)
    smooth_tcp_all = np.zeros((N, 3), dtype=np.float32)

    q_curr = ik_solver.q_home.copy()
    smooth_pos: Optional[np.ndarray] = None
    smooth_z: Optional[np.ndarray] = None

    for idx in range(N):
        if not det[idx]:
            smooth_pos = None
            smooth_z = None
            q_curr = ik_solver.q_home.copy()
            q_joints_all[idx] = q_curr
            continue

        pos_acc = tcp_w[idx].copy()
        z_acc = z_w[idx].copy()
        w_sum = 1.0
        for k_off, wk in [(1, 0.45), (2, 0.20)]:
            if idx + k_off < N and det[idx + k_off]:
                pos_acc += wk * tcp_w[idx + k_off]
                z_acc += wk * z_w[idx + k_off]
                w_sum += wk
        pos_avg = pos_acc / w_sum
        z_avg = z_acc / (float(np.linalg.norm(z_acc)) + 1e-8)

        if smooth_pos is None or smooth_z is None:
            smooth_pos = pos_avg
            smooth_z = z_avg
        else:
            smooth_pos = 0.55 * pos_avg + 0.45 * smooth_pos
            smooth_z = 0.45 * z_avg + 0.55 * smooth_z
            smooth_z /= float(np.linalg.norm(smooth_z)) + 1e-8

        R_target = build_hand_rot_matrix(smooth_z)
        q_curr = ik_solver.solve(smooth_pos, R_target, q_curr)
        ik_solver.d.qpos[:7] = q_curr
        mujoco.mj_forward(ik_solver.m, ik_solver.d)
        q_joints_all[idx] = q_curr.astype(np.float32)
        smooth_tcp_all[idx] = ik_solver.d.site_xpos[ik_solver.tcp_id].astype(np.float32)

    return det, tcp_2d, q_joints_all, smooth_tcp_all


def find_pick_and_place_episodes(
    traj: Dict[str, np.ndarray],
    det_bridged: np.ndarray,
    tcp_2d_bridged: np.ndarray,
    max_episodes: int = 25,
) -> List[Dict[str, int]]:
    """
    Find forward pick-and-place episodes (including the reaching motion, excluding cup resets):
    - Starts at the retracted hand peak after the cup is stationary at its initial table pose.
    - Reaches toward the cup with an open gripper.
    - Gripper closes onto the cup during grasp & transport.
    - Ends when the cup is placed on top of the pink box.
    """
    cup_u = traj["cup_2d"][:, 0]
    cup_v = traj["cup_2d"][:, 1]
    N = len(cup_u)
    valid_c = cup_u > 0
    idx_all = np.arange(N)
    cup_u_s = median_filter(np.interp(idx_all, idx_all[valid_c], cup_u[valid_c]), size=7)
    cup_v_s = median_filter(np.interp(idx_all, idx_all[valid_c], cup_v[valid_c]), size=7)

    du = np.zeros_like(cup_u_s)
    dv = np.zeros_like(cup_v_s)
    du[3:-3] = cup_u_s[6:] - cup_u_s[:-6]
    dv[3:-3] = cup_v_s[6:] - cup_v_s[:-6]
    speed = np.hypot(du, dv)
    dist = np.hypot(tcp_2d_bridged[:, 0] - cup_u_s, tcp_2d_bridged[:, 1] - cup_v_s)

    episodes: List[Dict[str, int]] = []
    f = 165
    while f < N - 30 and len(episodes) < max_episodes:
        if cup_u_s[f] > 1180.0 and cup_u_s[f + 1] <= 1180.0:
            window = cup_u_s[f : min(N, f + 150)]
            if np.min(window) < 1070.0:
                reach_box_rel = int(np.where(window < 1070.0)[0][0])
                f_cross = f + reach_box_rel

                f_move = f
                while f_move > 165 and speed[f_move] > 4.0:
                    f_move -= 1
                u_rest, v_rest = cup_u_s[f_move], cup_v_s[f_move]

                # Walk backward while the cup is stationary at its initial pose (within 25px) and hand is detected
                f_search_min = f_move
                while f_search_min > max(165, f_move - 75):
                    prev_f = f_search_min - 1
                    cup_still = (
                        np.hypot(cup_u_s[prev_f] - u_rest, cup_v_s[prev_f] - v_rest)
                        < 25.0
                    )
                    if cup_still and det_bridged[prev_f]:
                        f_search_min -= 1
                    else:
                        break

                # Start episode at the peak retracted distance before reaching toward the cup
                f_ep_start = f_search_min + int(
                    np.argmax(dist[f_search_min : f_move + 1])
                )

                # Exact frame when cup starts lifting/moving from its initial pose
                u_init, v_init = cup_u_s[f_ep_start], cup_v_s[f_ep_start]
                f_contact = f_move
                for fc in range(f_ep_start, f_cross):
                    if np.hypot(cup_u_s[fc] - u_init, cup_v_s[fc] - v_init) >= 12.0:
                        f_contact = fc
                        break

                # Gripper closes onto the cup 5 frames before the cup lifts off the table
                f_grasp = max(f_ep_start + 8, f_contact - 5)

                # Cup settles on top of the pink box
                f_settle = f_cross
                while f_settle < min(N - 1, f_cross + 60) and speed[f_settle] > 3.5:
                    f_settle += 1

                # Episode ends 8 frames after cup settles on the pink box (gripper releases)
                f_ep_end = min(N - 1, f_settle + 8)
                while f_ep_end > f_settle and not det_bridged[f_ep_end]:
                    f_ep_end -= 1

                reach_frames = f_contact - f_ep_start
                reach_disp = float(
                    np.linalg.norm(
                        tcp_2d_bridged[f_contact] - tcp_2d_bridged[f_ep_start]
                    )
                )
                if (
                    float(np.mean(det_bridged[f_ep_start : f_ep_end + 1])) == 1.0
                    and reach_frames >= 12
                    and reach_disp >= 80.0
                ):
                    episodes.append(
                        {
                            "start_frame": int(f_ep_start),
                            "grasp_frame": int(f_grasp),
                            "release_frame": int(f_settle),
                            "end_frame": int(f_ep_end),
                            "reach_frames": int(reach_frames),
                        }
                    )

                f = f_ep_end + 10
                while f < N - 10 and not (cup_u_s[f] > 1180.0 and speed[f] < 4.0):
                    f += 1
                continue
        f += 1

    return episodes


def find_push_episodes(
    traj: Dict[str, np.ndarray],
    det_bridged: np.ndarray,
    tcp_2d_bridged: np.ndarray,
    max_episodes: int = 25,
) -> List[Dict[str, int]]:
    """
    Find forward pushing episodes (including the reaching motion, excluding cup resets):
    - Starts at the retracted hand peak after the cup is stationary at its initial table pose.
    - Reaches forward to contact the cup and pushes it next to the pink box.
    - Stops right when it finishes pushing the cup.
    """
    cup_u = traj["cup_2d"][:, 0]
    cup_v = traj["cup_2d"][:, 1]
    qf = traj["qf"]
    N = len(cup_u)
    valid_c = cup_u > 0
    idx_all = np.arange(N)
    cup_u_s = median_filter(np.interp(idx_all, idx_all[valid_c], cup_u[valid_c]), size=7)
    cup_v_s = median_filter(np.interp(idx_all, idx_all[valid_c], cup_v[valid_c]), size=7)

    du = np.zeros_like(cup_u_s)
    dv = np.zeros_like(cup_v_s)
    du[3:-3] = cup_u_s[6:] - cup_u_s[:-6]
    dv[3:-3] = cup_v_s[6:] - cup_v_s[:-6]
    speed = np.hypot(du, dv)
    dist = np.hypot(tcp_2d_bridged[:, 0] - cup_u_s, tcp_2d_bridged[:, 1] - cup_v_s)

    episodes: List[Dict[str, int]] = []
    f = 10
    while f < N - 30 and len(episodes) < max_episodes:
        if cup_v_s[f] > 600.0 and cup_v_s[f + 1] <= 600.0:
            window = cup_v_s[f : min(N, f + 120)]
            if np.min(window) < 490.0:
                reach_rel = int(np.where(window < 490.0)[0][0])
                f_cross = f + reach_rel

                f_rest_peak = max(5, f - 45) + int(
                    np.argmax(cup_v_s[max(5, f - 45) : f + 1])
                )
                u_rest, v_rest = cup_u_s[f_rest_peak], cup_v_s[f_rest_peak]

                # Exact contact frame when cup begins sliding northward
                f_contact = f_rest_peak
                for fc in range(f_rest_peak, f_cross):
                    if v_rest - cup_v_s[fc] >= 10.0:
                        f_contact = fc
                        break

                # Walk backward while the cup is stationary at (u_rest, v_rest), hand is detected, and fingers are closed
                f_search_min = f_contact
                while f_search_min > max(5, f_contact - 65):
                    prev_f = f_search_min - 1
                    cup_still = (
                        np.hypot(cup_u_s[prev_f] - u_rest, cup_v_s[prev_f] - v_rest)
                        < 25.0
                    )
                    if cup_still and det_bridged[prev_f] and qf[prev_f] < 0.025:
                        f_search_min -= 1
                    else:
                        break

                # Start episode at the peak retracted distance before reaching to push
                f_ep_start = f_search_min + int(
                    np.argmax(dist[f_search_min : f_contact + 1])
                )

                # Cup finishes being pushed next to the pink box
                f_settle = f_cross
                while f_settle < min(N - 1, f_cross + 60) and speed[f_settle] > 3.5:
                    f_settle += 1

                f_ep_end = min(N - 1, f_settle + 4)
                while f_ep_end > f_settle and not det_bridged[f_ep_end]:
                    f_ep_end -= 1

                reach_frames = f_contact - f_ep_start
                if (
                    float(np.mean(det_bridged[f_ep_start : f_ep_end + 1])) == 1.0
                    and float(np.max(qf[f_ep_start : f_ep_end + 1])) < 0.025
                    and reach_frames >= 8
                ):
                    episodes.append(
                        {
                            "start_frame": int(f_ep_start),
                            "push_frame": int(f_contact),
                            "settle_frame": int(f_settle),
                            "end_frame": int(f_ep_end),
                            "reach_frames": int(reach_frames),
                        }
                    )

                f = f_ep_end + 10
                while f < N - 10 and not (cup_v_s[f] > 600.0 and speed[f] < 4.0):
                    f += 1
                continue
        f += 1

    return episodes


def process_task_into_dataset(
    overlay_video_path: str,
    episodes: List[Dict[str, int]],
    tcp_pos_all: np.ndarray,
    q_joints_all: np.ndarray,
    task_name: str,
    task_prompt: str,
    out_preview_video_path: str,
    dataset: LeRobotDataset,
    fps: int = 30,
) -> List[Dict[str, Any]]:
    """
    Read the overlaid video sequentially, crop out the ArUco marker (x >= 580),
    resize to 640x480, write the standalone split-episodes video, and populate
    the LeRobotDataset with 4D EEF state/action [tcp_x, tcp_y, tcp_z, gripper].
    """
    cap = cv2.VideoCapture(overlay_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open overlaid video: {overlay_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    preview_writer = cv2.VideoWriter(
        out_preview_video_path, fourcc, float(fps), (TARGET_WIDTH, TARGET_HEIGHT)
    )

    max_needed_frame = max(ep["end_frame"] for ep in episodes)
    # Map frame index -> episode index (episodes are non-overlapping and sorted)
    ep_idx = 0
    curr_ep = episodes[ep_idx]
    ep_frames_rgb: List[np.ndarray] = []

    summary_records: List[Dict[str, Any]] = []

    for f_idx in range(max_needed_frame + 1):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if f_idx < curr_ep["start_frame"]:
            continue

        # 0) Ignore left part of video, completely removing the ArUco marker
        cropped_bgr = frame_bgr[:, CROP_X_START:]
        resized_bgr = cv2.resize(
            cropped_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA
        )
        preview_writer.write(resized_bgr)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        ep_frames_rgb.append(resized_rgb)

        if f_idx == curr_ep["end_frame"]:
            s_f = curr_ep["start_frame"]
            e_f = curr_ep["end_frame"]
            T = e_f - s_f + 1

            # Build 4D End-Effector state [tcp_x, tcp_y, tcp_z, gripper] for every frame
            states_ep = np.zeros((T, 4), dtype=np.float32)
            states_ep[:, :3] = tcp_pos_all[s_f : e_f + 1]

            if task_name == "pick_and_place":
                g_f = curr_ep["grasp_frame"]
                r_f = curr_ep["release_frame"]
                for t_local in range(T):
                    f_abs = s_f + t_local
                    if g_f <= f_abs < r_f:
                        states_ep[t_local, 3] = GRIPPER_CLOSED
                    else:
                        states_ep[t_local, 3] = GRIPPER_OPEN
            else:
                # In pushing, the gripper stays closed while pushing the cup
                states_ep[:, 3] = GRIPPER_CLOSED

            # Next-step target actions: action[t] = state[min(t + 1, T - 1)]
            actions_ep = np.zeros_like(states_ep)
            actions_ep[:-1] = states_ep[1:]
            actions_ep[-1] = states_ep[-1]

            global_ep_index = dataset.meta.total_episodes
            for t_local in range(T):
                dataset.add_frame(
                    {
                        "observation.images.camera1": ep_frames_rgb[t_local],
                        "observation.state": states_ep[t_local],
                        "action": actions_ep[t_local],
                        "task": task_prompt,
                    }
                )
            dataset.save_episode()

            summary_records.append(
                {
                    "dataset_episode_index": int(global_ep_index),
                    "task_name": task_name,
                    "task_prompt": task_prompt,
                    "source_video": overlay_video_path,
                    "start_frame": int(s_f),
                    "end_frame": int(e_f),
                    "num_frames": int(T),
                    "initial_state": states_ep[0].tolist(),
                    "final_state": states_ep[-1].tolist(),
                    "initial_joints": q_joints_all[s_f].tolist(),
                    "final_joints": q_joints_all[e_f].tolist(),
                    **{
                        k: int(v)
                        for k, v in curr_ep.items()
                        if k not in ("start_frame", "end_frame")
                    },
                }
            )
            print(
                f"[{task_name}] Saved Episode {ep_idx + 1:02d}/{len(episodes)} "
                f"(global ep {global_ep_index:02d}): source frames [{s_f}..{e_f}] ({T} frames)",
                flush=True,
            )

            ep_frames_rgb = []
            ep_idx += 1
            if ep_idx >= len(episodes):
                break
            curr_ep = episodes[ep_idx]

    preview_writer.release()
    cap.release()
    print(f"Saved split-episodes preview video to: {out_preview_video_path}", flush=True)
    return summary_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split overlaid Franka videos into 25 episodes per task and build a LeRobot dataset for SmolVLA."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="lerobot_dataset",
        help="Output directory for the LeRobot v3.0 dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="humanoid_challenge/franka_cup_smolvla",
        help="Repository ID for the LeRobotDataset.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=25,
        help="Number of episodes per task (default: 25).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Directory for optional trajectory cache.",
    )
    args = parser.parse_args()

    t0 = time.time()
    cal = np.load("camera_extrinsics.npz")
    K = cal["K"]
    R_wc = cal["R_marker_cam"]
    t_wc = cal["t_marker_cam"].ravel()
    fy = float(K[1, 1])

    fr3_xml = build_fr3_xml("franka_fr3/fr3.xml", R_wc, t_wc, fy, img_h=1080)
    model = mujoco.MjModel.from_xml_string(fr3_xml)
    data = mujoco.MjData(model)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_hand")
    ik_solver = FrankaIKSolver(model, data, tcp_id, hand_id)

    cache_pick = (
        os.path.join(args.cache_dir, "traj_pick.npz") if args.cache_dir else None
    )
    cache_push = (
        os.path.join(args.cache_dir, "traj_push.npz") if args.cache_dir else None
    )

    print("1. Extracting/loading trajectories and solving full EEF & joint states...", flush=True)
    traj_pick = extract_or_load_raw_trajectory(
        "raw_pick_and_place.mp4",
        "camera_extrinsics.npz",
        min_active_frame=140,
        cache_npz_path=cache_pick,
    )
    det_pick, tcp_2d_pick, q_joints_pick, tcp_pos_pick = compute_full_joint_trajectory(
        traj_pick, ik_solver
    )
    pick_episodes = find_pick_and_place_episodes(
        traj_pick, det_pick, tcp_2d_pick, max_episodes=args.episodes_per_task
    )

    traj_push = extract_or_load_raw_trajectory(
        "raw_push.mp4",
        "camera_extrinsics.npz",
        min_active_frame=0,
        cache_npz_path=cache_push,
    )
    det_push, tcp_2d_push, q_joints_push, tcp_pos_push = compute_full_joint_trajectory(
        traj_push, ik_solver
    )
    push_episodes = find_push_episodes(
        traj_push, det_push, tcp_2d_push, max_episodes=args.episodes_per_task
    )

    if len(pick_episodes) < args.episodes_per_task:
        raise RuntimeError(
            f"Expected {args.episodes_per_task} pick_and_place episodes, found {len(pick_episodes)}"
        )
    if len(push_episodes) < args.episodes_per_task:
        raise RuntimeError(
            f"Expected {args.episodes_per_task} push episodes, found {len(push_episodes)}"
        )

    dataset_root = Path(args.dataset_root)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (TARGET_HEIGHT, TARGET_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": [
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": [
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "gripper",
            ],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=30,
        features=features,
        root=dataset_root,
        robot_type="franka_fr3",
        use_videos=True,
        vcodec="h264",
        streaming_encoding=True,
    )

    all_summaries: List[Dict[str, Any]] = []
    all_summaries.extend(
        process_task_into_dataset(
            overlay_video_path="franka_overlay_pick_and_place.mp4",
            episodes=pick_episodes,
            tcp_pos_all=tcp_pos_pick,
            q_joints_all=q_joints_pick,
            task_name="pick_and_place",
            task_prompt=PROMPT_PICK_AND_PLACE,
            out_preview_video_path="dataset_pick_and_place_25eps.mp4",
            dataset=dataset,
            fps=30,
        )
    )
    all_summaries.extend(
        process_task_into_dataset(
            overlay_video_path="franka_overlay_push.mp4",
            episodes=push_episodes,
            tcp_pos_all=tcp_pos_push,
            q_joints_all=q_joints_push,
            task_name="push",
            task_prompt=PROMPT_PUSH,
            out_preview_video_path="dataset_push_25eps.mp4",
            dataset=dataset,
            fps=30,
        )
    )

    dataset.finalize()

    summary_path = dataset_root / "episodes_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_id": args.repo_id,
                "state_space": "4D_eef_pos_gripper",
                "state_names": ["tcp_x", "tcp_y", "tcp_z", "gripper"],
                "total_episodes": len(all_summaries),
                "total_frames": int(dataset.meta.total_frames),
                "crop_x_start": CROP_X_START,
                "resolution": [TARGET_WIDTH, TARGET_HEIGHT],
                "gripper_open_value": GRIPPER_OPEN,
                "gripper_closed_value": GRIPPER_CLOSED,
                "tasks": {
                    "pick_and_place": PROMPT_PICK_AND_PLACE,
                    "push": PROMPT_PUSH,
                },
                "episodes": all_summaries,
            },
            f,
            indent=2,
        )

    print(
        f"Done in {time.time() - t0:.1f}s! "
        f"Created LeRobotDataset at '{dataset_root}' with {len(all_summaries)} episodes "
        f"({dataset.meta.total_frames} total frames).",
        flush=True,
    )


if __name__ == "__main__":
    main()

