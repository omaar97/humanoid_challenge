#!/usr/bin/env python3
"""
Trajectory Viewer & Closed-Loop Executor for SmolVLA in MuJoCo.

Visualizes and executes the multi-step trajectory chunk predicted by the fine-tuned SmolVLA policy:
- Queries the policy using `predict_action_chunk(batch)` to retrieve the full 50-step future action horizon.
- Strictly loads the real SmolVLA checkpoint (raises FileNotFoundError immediately if missing — NO mock fallbacks).
- Language Prompt Support: Allows custom language prompts via `--prompt` to test instruction following.
- Phase 1 (Static Inspection): Holds the robot stationary for a configurable duration (`--static-sec`) while
  displaying the predicted 3D trajectory markers in the interactive viewer for user inspection and camera manipulation.
- Phase 2 (Slow Execution & Replanning): Executes the trajectory closed-loop (`--steps`, `--step-delay`, `--replan-interval`),
  continually re-querying SmolVLA and updating the 3D trajectory ribbon in real time so the user sees the arm move
  and the predicted path adapt dynamically.
- Interactive 3D GUI: Opens `mujoco.viewer.launch_passive` with orbit, pan, zoom, and live 3D spline markers.
- Dual-View Recording: Saves synchronized 2D projected trajectory + 3D overview video (`smolvla_predicted_trajectories.mp4`).
"""

import argparse
import os
import sys
import time
from pathlib import Path
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
import torch

# Robust root path resolution: works whether invoked from project root or src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mujoco_test_env import (
    CROP_X_START,
    PROMPT_PICK_AND_PLACE,
    PROMPT_PUSH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TEST_INSTANCES,
    FrankaCupMuJoCoEnv,
    resolve_pretrained_model_dir,
)

# Strictly the 2 canonical prompt choices corresponding to the 2 dataset tasks
PROMPT_CHOICES_HELP = {
    "pick_and_place": PROMPT_PICK_AND_PLACE,
    "push": PROMPT_PUSH,
}


def resolve_prompt(prompt_input: Optional[str], default_task: str = "pick_and_place") -> Tuple[str, str]:
    """
    Resolve user prompt input to one of the 2 valid canonical dataset task prompts.
    Returns: (canonical_prompt_str, canonical_task_name)
    """
    if not prompt_input or not prompt_input.strip():
        chosen = default_task
    else:
        chosen = prompt_input.strip()

    if "push" in chosen.lower():
        return PROMPT_PUSH, "push"
    else:
        return PROMPT_PICK_AND_PLACE, "pick_and_place"


def resolve_file(rel_path: str) -> str:
    """Resolve a relative path against CWD first, then REPO_ROOT."""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    cwd_path = Path.cwd() / rel_path
    if cwd_path.exists():
        return str(cwd_path.resolve())
    root_path = REPO_ROOT / rel_path
    if root_path.exists():
        return str(root_path.resolve())
    return str(cwd_path.resolve())


def load_smolvla_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[Any, Any, Any, str]:
    """
    Strictly load the fine-tuned SmolVLA policy from disk.
    Raises FileNotFoundError immediately if the checkpoint cannot be resolved.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if checkpoint_path is None:
        for cand in ["last", "pretrained_model"]:
            resolved = resolve_pretrained_model_dir(cand)
            if os.path.isdir(resolved) and os.path.isfile(os.path.join(resolved, "model.safetensors")):
                checkpoint_path = cand
                break

    if checkpoint_path is None:
        raise FileNotFoundError(
            "SmolVLA checkpoint not found! Please specify a valid checkpoint folder with --checkpoint <path>."
        )

    resolved_ckpt = resolve_pretrained_model_dir(checkpoint_path)
    weights_path = os.path.join(resolved_ckpt, "model.safetensors")
    if not (os.path.isdir(resolved_ckpt) and os.path.isfile(weights_path)):
        raise FileNotFoundError(
            f"Cannot find valid SmolVLA model at '{resolved_ckpt}' (missing 'model.safetensors')."
        )

    print(f"\n[Loading REAL SmolVLA policy from: {resolved_ckpt} on {device} ...]", flush=True)

    tok_dir = os.path.join(resolved_ckpt, "tokenizer")
    pre_overrides = {"device_processor": {"device": device}}
    if os.path.isdir(tok_dir):
        pre_overrides["tokenizer_processor"] = {"tokenizer_name": tok_dir}

    policy = SmolVLAPolicy.from_pretrained(resolved_ckpt)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=resolved_ckpt,
        preprocessor_overrides=pre_overrides,
    )
    return policy, preprocessor, postprocessor, resolved_ckpt


def query_current_predicted_trajectory(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env: FrankaCupMuJoCoEnv,
) -> np.ndarray:
    """
    Query SmolVLA to predict the multi-step trajectory chunk (all 50 future waypoints).
    Does NOT reset the policy queue so it can safely be queried alongside select_action.
    """

    obs = env.get_observation()
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
        actions_t = policy.predict_action_chunk(batch)
        actions_unnorm = postprocessor(actions_t)
        chunk_np = actions_unnorm.squeeze(0).cpu().numpy()

    return chunk_np


def project_points_to_camera(
    points_3d: np.ndarray,
    calib_path: str = "camera_extrinsics.npz",
    img_w: int = 1920,
    img_h: int = 1080,
) -> List[Tuple[int, int]]:
    """Project 3D world waypoints onto the calibrated 1920x1080 camera frame, then crop/scale to 640x480."""
    cal_file = resolve_file(calib_path)
    cal = np.load(cal_file)
    K = cal["K"]
    R_cam_marker = cal["R_cam_marker"]
    t_cam_marker = cal["t_cam_marker"].ravel()

    pts_2d = []
    scale_x = float(TARGET_WIDTH) / float(img_w - CROP_X_START)
    scale_y = float(TARGET_HEIGHT) / float(img_h)

    for pt in points_3d:
        p_c = (R_cam_marker @ np.asarray(pt, dtype=np.float64).ravel()).ravel() + t_cam_marker
        if float(p_c[2]) <= 0.05:
            continue
        u = float(K[0, 0] * p_c[0] / p_c[2] + K[0, 2])
        v = float(K[1, 1] * p_c[1] / p_c[2] + K[1, 2])

        u_crop = (u - CROP_X_START) * scale_x
        v_crop = v * scale_y
        pts_2d.append((int(round(u_crop)), int(round(v_crop))))

    return pts_2d


def draw_single_trajectory_on_image(
    img_bgr: np.ndarray,
    pts_2d: List[Tuple[int, int]],
    color_start: Tuple[int, int, int] = (0, 255, 255),  # Yellow
    color_end: Tuple[int, int, int] = (255, 0, 255),    # Magenta
    title: str = "Predicted SmolVLA Trajectory",
) -> np.ndarray:
    """Draw the single predicted trajectory ribbon and waypoints on the 640x480 camera view."""
    out = img_bgr.copy()
    n = len(pts_2d)
    if n < 2:
        return out

    # Draw continuous path line with color gradient
    for i in range(n - 1):
        pt1 = pts_2d[i]
        pt2 = pts_2d[i + 1]
        alpha = float(i) / float(n - 1)
        b = int((1.0 - alpha) * color_start[0] + alpha * color_end[0])
        g = int((1.0 - alpha) * color_start[1] + alpha * color_end[1])
        r = int((1.0 - alpha) * color_start[2] + alpha * color_end[2])
        cv2.line(out, pt1, pt2, (b, g, r), 3, cv2.LINE_AA)

    # Draw waypoint circles
    for i, pt in enumerate(pts_2d):
        alpha = float(i) / float(n - 1)
        b = int((1.0 - alpha) * color_start[0] + alpha * color_end[0])
        g = int((1.0 - alpha) * color_start[1] + alpha * color_end[1])
        r = int((1.0 - alpha) * color_start[2] + alpha * color_end[2])
        radius = 5 if (i == 0 or i == n - 1) else 3
        cv2.circle(out, pt, radius, (b, g, r), -1, cv2.LINE_AA)
        cv2.circle(out, pt, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Labels for start and horizon goal
    cv2.putText(
        out,
        "Start",
        (pts_2d[0][0] + 8, pts_2d[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Goal (t+{n})",
        (pts_2d[-1][0] + 8, pts_2d[-1][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        title,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def add_user_scene_markers(
    viewer: Any,
    waypoints_3d: np.ndarray,
    color_rgba: Tuple[float, float, float, float] = (1.0, 0.8, 0.1, 0.85),
) -> None:
    """Inject connected 3D tube and waypoint sphere markers into the interactive MuJoCo passive viewer scene."""
    if viewer is None or not hasattr(viewer, "user_scn") or viewer.user_scn is None:
        return
    scn = viewer.user_scn
    scn.ngeom = 0
    n = len(waypoints_3d)
    if n == 0:
        return

    # 1. Connect consecutive waypoints with 3D capsule tubes
    for i in range(n - 1):
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        p1 = np.asarray(waypoints_3d[i], dtype=np.float64)
        p2 = np.asarray(waypoints_3d[i + 1], dtype=np.float64)
        alpha = float(i) / float(max(1, n - 1))
        # Tube color transitions from gold to orange
        r = float((1.0 - alpha) * 1.0 + alpha * 1.0)
        g = float((1.0 - alpha) * 0.82 + alpha * 0.25)
        b = float((1.0 - alpha) * 0.15 + alpha * 0.85)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0022, p1, p2)
        geom.rgba = [r, g, b, 0.80]
        scn.ngeom += 1

    # 2. Add distinct spheres at waypoints: Green (Start), Gold/Orange (Intermediate), Magenta (Horizon Goal)
    for i, pt in enumerate(waypoints_3d):
        if scn.ngeom >= scn.maxgeom:
            break
        alpha = float(i) / float(max(1, n - 1))
        if i == 0:
            rgba = [0.10, 0.95, 0.25, 0.95]
            radius = 0.0085  # Prominent start sphere
        elif i == n - 1:
            rgba = [0.95, 0.15, 0.95, 0.95]
            radius = 0.0090  # Prominent goal horizon sphere
        else:
            r = float((1.0 - alpha) * 1.0 + alpha * 1.0)
            g = float((1.0 - alpha) * 0.82 + alpha * 0.25)
            b = float((1.0 - alpha) * 0.15 + alpha * 0.85)
            rgba = [r, g, b, 0.85]
            radius = 0.0042

        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0.0, 0.0],
            pos=np.asarray(pt, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        scn.ngeom += 1


def run_trajectory_viewer(
    checkpoint_path: Optional[str] = None,
    task_name: str = "pick_and_place",
    instance_idx: int = 0,
    prompt: Optional[str] = None,
    static_sec: float = 3.0,
    max_steps: int = 150,
    step_delay: float = 0.05,
    replan_interval: int = 1,
    output_video_path: str = "smolvla_predicted_trajectories.mp4",
    save_video: bool = True,
    device: str = "cuda",
    hold_at_end: bool = True,
) -> None:
    """
    Visualizes and executes the predicted SmolVLA trajectory:
    1. Static phase: holds robot motionless at home posture for `static_sec` seconds while rendering the
       initial predicted 3D trajectory markers in MuJoCo viewer (user can orbit/pan/zoom).
    2. Slow execution phase: executes up to `max_steps` closed-loop steps with delay `step_delay`,
       continually updating the 3D trajectory in real time.
    """
    calib_file = resolve_file("camera_extrinsics.npz")
    fr3_file = resolve_file("franka_fr3/fr3.xml")

    # Strictly load the real model — raises FileNotFoundError immediately if missing
    policy, preprocessor, postprocessor, resolved_dir = load_smolvla_model(
        checkpoint_path=checkpoint_path, device=device
    )

    env = FrankaCupMuJoCoEnv(fr3_xml_path=fr3_file, calib_path=calib_file)

    # Resolve active task prompt to one of the 2 canonical dataset task prompts
    active_prompt, resolved_task = resolve_prompt(prompt, default_task=task_name)
    if (prompt is not None and prompt.strip()) and resolved_task != task_name:
        task_name = resolved_task
    inst = TEST_INSTANCES[task_name][instance_idx]

    is_push = (task_name == "push")
    cx, cy = float(inst["cup_xy"][0]), float(inst["cup_xy"][1])

    if is_push:
        init_tcp = np.array([cx, cy - 0.075, 0.048], dtype=np.float64)
        init_grip = 0.0  # closed gripper
    else:
        init_tcp = np.array([0.385, -0.105, 0.048], dtype=np.float64)
        init_grip = 0.04  # open gripper

    obs = env.reset(
        cup_xy=inst["cup_xy"],
        init_tcp_pos=init_tcp,
        init_gripper=init_grip,
        task_prompt=active_prompt,
    )

    print("\n" + "=" * 72)
    print("  SMOLVLA TRAJECTORY VIEWER & CLOSED-LOOP EXECUTOR")
    print("=" * 72)
    print(f"  EXECUTING PROMPT: '{active_prompt}'")
    print(f"  TASK SCENARIO:    {task_name.upper()} (Instance #{instance_idx}: '{inst['name']}')")
    print("=" * 72 + "\n", flush=True)

    # Launch interactive MuJoCo passive viewer if display is available
    viewer = None
    has_gui = False
    try:
        if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            has_gui = viewer.is_running()
            if has_gui:
                print("\n" + "=" * 68)
                print("  INTERACTIVE MUJOCO 3D VIEWER OPENED SUCCESSFULLY!")
                print("  - Left-click & drag: Orbit camera around the scene")
                print("  - Right-click & drag: Pan camera")
                print("  - Scroll wheel: Zoom in / out")
                print("  - Close the viewer window or press ESC when done")
                print("=" * 68 + "\n", flush=True)
    except Exception as e:
        print(f"Interactive viewer GUI not available ({e}). Defaulting to video output.", flush=True)
        has_gui = False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    video_writer = None
    if save_video or not has_gui:
        video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            24.0,
            (TARGET_WIDTH * 2, TARGET_HEIGHT),
        )
        print(f"Recording evaluation video to: {output_video_path} ...", flush=True)

    def write_video_frame(left_title: str, phase_tag: str, cur_waypoints_3d: np.ndarray) -> None:
        if video_writer is None:
            return
        smolvla_rgb = env.render_smolvla_input()
        overview_rgb = cv2.resize(
            env.render_overview("overview_cam"),
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        left_bgr = cv2.cvtColor(smolvla_rgb, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2BGR)
        pts_2d = project_points_to_camera(cur_waypoints_3d, calib_file)
        annotated_left = draw_single_trajectory_on_image(left_bgr, pts_2d, title=left_title)

        # 3rd-person overview annotations
        cv2.putText(
            right_bgr,
            "3rd-Person Overview Camera",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            right_bgr,
            f"Phase: {phase_tag}",
            (14, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (100, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            right_bgr,
            f"Prompt: {active_prompt}",
            (14, 462),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )
        video_writer.write(np.hstack([annotated_left, right_bgr]))

    # -------------------------------------------------------------------------
    # PHASE 1: Static Inspection Phase (Continuous Noise Sampling)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  [Phase 1/2] STATIC INSPECTION PHASE")
    print(f"  EXECUTING PROMPT: '{active_prompt}'")
    print(f"  Robot static at home posture for {static_sec:.1f}s while scene is static.")
    print("  Continuously re-predicting 3D trajectory to visualize Flow Matching")
    print("  action generation noise in real time:")
    print("    - Green sphere: Start (t=0)")
    print("    - Gold/Orange tube: Predicted trajectory horizon (50 steps)")
    print("    - Magenta sphere: Goal horizon (t=50)")
    print("=" * 72 + "\n", flush=True)

    t_start = time.time()
    noise_sample_idx = 0
    waypoints_3d = np.zeros((50, 3), dtype=np.float64)

    # Continuously sample and display new trajectory variants to show policy noise
    while (time.time() - t_start) < static_sec:
        if viewer is not None and not viewer.is_running():
            print("Viewer window closed by user during static pause.", flush=True)
            break

        # Re-query policy with fresh Gaussian noise vector from flow matching prior
        pred_traj_4d = query_current_predicted_trajectory(policy, preprocessor, postprocessor, env)
        waypoints_3d = pred_traj_4d[:, :3]

        if has_gui and viewer is not None:
            add_user_scene_markers(viewer, waypoints_3d)
            viewer.sync()

        elapsed = time.time() - t_start
        rem = max(0.0, static_sec - elapsed)

        if video_writer is not None:
            write_video_frame(
                left_title=f"Static Action Noise | Sample #{noise_sample_idx+1} ({rem:.1f}s rem)",
                phase_tag=f"STATIC ACTION NOISE ({rem:.1f}s)",
                cur_waypoints_3d=waypoints_3d,
            )

        noise_sample_idx += 1
        time.sleep(max(0.04, step_delay))

    # Reset policy once before physical execution begins so action queues are clean
    policy.reset()

    # -------------------------------------------------------------------------
    # PHASE 2: Slow Closed-Loop Trajectory Execution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  [Phase 2/2] CLOSED-LOOP TRAJECTORY EXECUTION")
    print(f"  EXECUTING PROMPT: '{active_prompt}'")
    print(f"  Executing up to {max_steps} steps slowly (delay={step_delay*1000:.0f}ms per step)...")
    print("  Executing action queue sequentially toward the cup while live 3D")
    print("  predicted trajectory updates in the viewer!")
    print("=" * 72 + "\n", flush=True)

    step_i = 0
    task_success = False

    while step_i < max_steps:
        if viewer is not None and not viewer.is_running():
            print("Viewer window closed by user. Terminating rollout.", flush=True)
            break

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
            # 1. Action to execute from LeRobot policy queue (ensures smooth forward chunk rollout)
            action_t = policy.select_action(batch)

            # 2. Predicted future trajectory horizon for 3D visualization
            if step_i % replan_interval == 0:
                vis_chunk_t = policy.predict_action_chunk(batch)
                vis_chunk_np = postprocessor(vis_chunk_t).squeeze(0).cpu().numpy()
                waypoints_3d = vis_chunk_np[:, :3]

        action_4d = postprocessor(action_t).squeeze(0).cpu().numpy()

        # Update 3D scene markers in viewer
        if has_gui and viewer is not None:
            add_user_scene_markers(viewer, waypoints_3d)
            viewer.sync()

        # Step robot physics using the next predicted waypoint
        obs, info = env.step(action_4d)

        # Write video frame
        if video_writer is not None:
            write_video_frame(
                left_title=f"Executing SmolVLA Trajectory | Step {step_i+1}/{max_steps}",
                phase_tag=f"EXECUTING (Step {step_i+1}/{max_steps})",
                cur_waypoints_3d=waypoints_3d,
            )

        # Artificial slowdown delay to make motion and updates clearly observable
        if step_delay > 0:
            time.sleep(step_delay)

        step_i += 1

        # Check task completion criteria
        if info.get("on_pink_box", False):
            print(f"\n>>> [SUCCESS] Cup successfully placed on pink box at step {step_i}! <<<", flush=True)
            task_success = True
            break
        elif info.get("next_to_pink_box", False):
            print(f"\n>>> [SUCCESS] Cup successfully pushed next to pink box at step {step_i}! <<<", flush=True)
            task_success = True
            break

    # -------------------------------------------------------------------------
    # Hold & Cleanup
    # -------------------------------------------------------------------------
    if video_writer is not None:
        video_writer.release()
        print(f"\nSaved evaluation video to: {output_video_path}", flush=True)

    if has_gui and viewer is not None and viewer.is_running() and hold_at_end:
        print("\n" + "=" * 68)
        print("  EXECUTION COMPLETE!")
        print("  Viewer is holding final posture for inspection.")
        print("  Close the viewer window or press ESC when done.")
        print("=" * 68 + "\n", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.04)

    if viewer is not None:
        viewer.close()
    env.close()
    print("Trajectory viewer finished successfully.", flush=True)


# Backward-compatible alias
run_trajectory_sweep = run_trajectory_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize and slowly execute predicted SmolVLA trajectories in MuJoCo viewer or video."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or HF repo_id of fine-tuned SmolVLA checkpoint (auto-detects local checkpoint if omitted).",
    )
    parser.add_argument(
        "--task",
        choices=["pick_and_place", "push"],
        default="pick_and_place",
        help="Task scenario (pick_and_place or push).",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="Test instance index (0 to 9).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=PROMPT_PICK_AND_PLACE,
        choices=[
            "pick_and_place",
            "push",
            PROMPT_PICK_AND_PLACE,
            PROMPT_PUSH,
            "pick up the green cup and place it on the pink box",
        ],
        help=(
            "Task language prompt (strictly 2 choices in the dataset):\n"
            f"  1) 'pick_and_place' (or '{PROMPT_PICK_AND_PLACE}') [DEFAULT]\n"
            f"  2) 'push'           (or '{PROMPT_PUSH}')"
        ),
    )
    parser.add_argument(
        "--static-sec",
        type=float,
        default=3.0,
        help="Duration in seconds to hold the robot static at the beginning before moving (default: 3.0s).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Maximum closed-loop execution steps (default: 150).",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.05,
        help="Delay in seconds per execution step to make motion and trajectory updates easily visible (default: 0.05s).",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=1,
        help="How frequently to query SmolVLA for a fresh trajectory chunk (default: 1 = every step).",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="smolvla_predicted_trajectories.mp4",
        help="Output video path when recording.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording when interactive GUI viewer is running.",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Exit immediately after rollout without keeping the viewer open for inspection.",
    )
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help="Device for SmolVLA inference (cuda or cpu).",
    )
    args = parser.parse_args()

    run_trajectory_viewer(
        checkpoint_path=args.checkpoint,
        task_name=args.task,
        instance_idx=args.instance,
        prompt=args.prompt,
        static_sec=args.static_sec,
        max_steps=args.steps,
        step_delay=args.step_delay,
        replan_interval=args.replan_interval,
        output_video_path=args.output_video,
        save_video=not args.no_video,
        device=args.device,
        hold_at_end=not args.no_hold,
    )


if __name__ == "__main__":
    main()
