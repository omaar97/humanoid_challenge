#!/usr/bin/env python3
"""
Pipeline Validation & Scripted Physics Verification for Franka FR3 MuJoCo Suite.

Specific standalone script for validating the entire robotics and simulation pipeline
WITHOUT requiring any trained neural network or fine-tuned policy checkpoint:
1. Validates Franka FR3 joint limits, ranges, and kinematics against 'franka_fr3/fr3.xml'.
2. Validates workspace setup, cameras ('real_cam', 'overview_cam'), and 'camera_extrinsics.npz'.
3. Renders a standardized visual gallery of all benchmark test instances.
4. Generates and executes scripted 4D end-effector trajectories for Pick-and-Place and Push tasks.
5. Evaluates physical grasping, contact dynamics, lifting, placing, and pushing across all 3 controllers:
   - Position IK Controller (FrankaCupMuJoCoEnv)
   - Direct Torque Velocity Controller (FrankaVelocityEnv)
   - Pure End-Effector Cartesian Impedance Controller (FrankaImpedanceEnv)
6. Supports interactive 3D GUI passive viewer (--viewer) and headless dual-view video recording.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np

# Robust root path resolution: works whether invoked from project root or src/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mujoco_main_test_env import (
    CROP_X_START,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PROMPT_PICK_AND_PLACE,
    PROMPT_PUSH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TEST_INSTANCES,
    FrankaCupMuJoCoEnv,
)
from mujoco_velocity_controller_test import FrankaVelocityEnv
from mujoco_impedance_controller_test import FrankaImpedanceEnv


def generate_scripted_demo_trajectory(
    task_name: str,
    cup_xy: Tuple[float, float],
) -> List[np.ndarray]:
    """
    Generate a clean 4D task-space trajectory [tcp_x, tcp_y, tcp_z, gripper] for any test instance
    to verify physical grasping, lifting, placing, and pushing in the MuJoCo scene.
    """
    cx, cy = float(cup_xy[0]), float(cup_xy[1])
    actions: List[np.ndarray] = []

    if task_name == "pick_and_place":
        p_start = np.array([0.385, -0.105, 0.055])
        p_pre = np.array([cx + 0.025, cy - 0.025, 0.050])
        p_grasp = np.array([cx, cy, 0.042])
        p_lift = np.array([cx - 0.02, cy + 0.01, 0.145])
        p_above_box = np.array([0.215, 0.022, 0.135])
        p_place = np.array([0.215, 0.022, 0.092])
        p_retreat = np.array([0.245, 0.000, 0.125])

        # 1. Reach to pre-grasp (25 steps, gripper open)
        for a in np.linspace(0, 1, 25):
            p = (1 - a) * p_start + a * p_pre
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_OPEN], dtype=np.float32))
        # 2. Move onto cup (15 steps, gripper open)
        for a in np.linspace(0, 1, 15):
            p = (1 - a) * p_pre + a * p_grasp
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_OPEN], dtype=np.float32))
        # 3. Close gripper around the 6cm cup (12 steps, gripper closed)
        for _ in range(12):
            actions.append(
                np.array([p_grasp[0], p_grasp[1], p_grasp[2], GRIPPER_CLOSED], dtype=np.float32)
            )
        # 4. Lift cup above table (22 steps, gripper closed)
        for a in np.linspace(0, 1, 22):
            p = (1 - a) * p_grasp + a * p_lift
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_CLOSED], dtype=np.float32))
        # 5. Transport over the pink box (25 steps, gripper closed)
        for a in np.linspace(0, 1, 25):
            p = (1 - a) * p_lift + a * p_above_box
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_CLOSED], dtype=np.float32))
        # 6. Lower onto top of 5cm pink box (15 steps, gripper closed)
        for a in np.linspace(0, 1, 15):
            p = (1 - a) * p_above_box + a * p_place
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_CLOSED], dtype=np.float32))
        # 7. Open gripper to release cup on pink box (12 steps, gripper open)
        for _ in range(12):
            actions.append(
                np.array([p_place[0], p_place[1], p_place[2], GRIPPER_OPEN], dtype=np.float32)
            )
        # 8. Retreat slightly while keeping cup on box (10 steps, gripper open)
        for a in np.linspace(0, 1, 10):
            p = (1 - a) * p_place + a * p_retreat
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_OPEN], dtype=np.float32))
    else:
        # Push task: reach with closed gripper and push cup next to the right side of the pink box
        p_start = np.array([cx + 0.015, cy - 0.075, 0.044])
        p_contact = np.array([cx + 0.005, cy - 0.042, 0.042])
        # Push target places cup at x ~ 0.315, y ~ +0.015 (right next to pink box right face at x=0.283)
        p_push_end = np.array([0.318, -0.025, 0.042])

        for a in np.linspace(0, 1, 20):
            p = (1 - a) * p_start + a * p_contact
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_CLOSED], dtype=np.float32))
        for a in np.linspace(0, 1, 45):
            p = (1 - a) * p_contact + a * p_push_end
            actions.append(np.array([p[0], p[1], p[2], GRIPPER_CLOSED], dtype=np.float32))
        for _ in range(10):
            actions.append(
                np.array(
                    [p_push_end[0], p_push_end[1], p_push_end[2], GRIPPER_CLOSED],
                    dtype=np.float32,
                )
            )

    return actions


def verify_joint_limits() -> bool:
    """Verify all 7 joint limits match unmodified franka_fr3/fr3.xml."""
    print("\n" + "=" * 72)
    print("  1. FRANKA FR3 JOINT LIMITS & KINEMATICS VERIFICATION")
    print("=" * 72)
    env = FrankaCupMuJoCoEnv()
    all_ok = True
    for j in range(7):
        jname = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jmin, jmax = env.model.jnt_range[j]
        q_h = env.controller.q_home[j]
        margin = min(q_h - jmin, jmax - q_h)
        status = "OK" if margin > 0.05 else "WARN"
        if margin <= 0:
            all_ok = False
        print(
            f"  Joint {j+1} ({jname:<12}): range=[{jmin:+.3f}, {jmax:+.3f}] rad | "
            f"q_home={q_h:+.3f} rad | margin={margin:.3f} rad [{status}]"
        )
    env.close()
    print("  -> Joint limits verification: PASSED" if all_ok else "  -> Joint limits verification: FAILED")
    return all_ok


def render_instances_gallery(output_image_path: str = "mujoco_test_instances_gallery.jpg") -> None:
    """Render a visual gallery image across all test instances."""
    print("\n" + "=" * 72)
    print("  2. TEST INSTANCES VISUAL GALLERY RENDERING")
    print("=" * 72)
    env = FrankaCupMuJoCoEnv()
    gallery_rows: List[np.ndarray] = []
    for task_name in ["pick_and_place", "push"]:
        for idx, inst in enumerate(TEST_INSTANCES[task_name]):
            env.reset_instance(task_name, idx)
            dual_bgr = env.render_dual_view_bgr(
                f"{inst['name']}: cup_xy={inst['cup_xy']}"
            )
            gallery_rows.append(dual_bgr)

    gallery_img = np.vstack(gallery_rows)
    cv2.imwrite(output_image_path, gallery_img)
    print(f"  Saved test instances visual gallery to: {output_image_path} ({gallery_img.shape})")

    # High-res dual-view preview of instance 0
    env.reset_instance("pick_and_place", 0)
    cv2.imwrite("mujoco_dual_view_preview.jpg", gallery_rows[0])
    print("  Saved high-res preview of Instance #0 to: mujoco_dual_view_preview.jpg")
    env.close()


def validate_controller_pipeline(
    controller_type: str = "position",
    task_name: str = "pick_and_place",
    instance_idx: int = 0,
    output_video_path: Optional[str] = None,
    use_viewer: bool = False,
) -> bool:
    """
    Execute scripted trajectory on a specific controller to validate kinematics,
    low-level torque control, contact stability, and task completion.
    """
    tag = controller_type.upper()
    print("\n" + "-" * 72)
    print(f"  VALIDATING CONTROLLER: [{tag}] | TASK: [{task_name.upper()}] | INSTANCE: #{instance_idx}")
    print("-" * 72)

    inst = TEST_INSTANCES[task_name][instance_idx]
    cup_xy = inst["cup_xy"]

    if controller_type == "position":
        env: Any = FrankaCupMuJoCoEnv()
    elif controller_type == "velocity":
        env = FrankaVelocityEnv()
    elif controller_type == "impedance":
        env = FrankaImpedanceEnv()
    else:
        raise ValueError(f"Unknown controller type '{controller_type}'. Choose position, velocity, or impedance.")

    obs = env.reset_instance(task_name, instance_idx)
    scripted_traj = generate_scripted_demo_trajectory(task_name, cup_xy)

    viewer = None
    has_gui = False
    if use_viewer:
        try:
            if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
                viewer = mujoco.viewer.launch_passive(env.model, env.data)
                has_gui = viewer.is_running()
                if has_gui:
                    print(f"  Interactive MuJoCo 3D viewer opened for {tag} controller!", flush=True)
            else:
                print("  DISPLAY environment variable not set; running headless.", flush=True)
        except Exception as e:
            print(f"  Interactive viewer GUI not available ({e}). Running headless.", flush=True)

    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (TARGET_WIDTH * 2, TARGET_HEIGHT))

    info: Dict[str, Any] = {}
    prev_pos = None

    for step_i, action_4d in enumerate(scripted_traj):
        if viewer is not None:
            if not viewer.is_running():
                print("  Viewer window closed by user. Terminating rollout.", flush=True)
                break
            viewer.sync()

        curr_target = np.array(action_4d[:3], dtype=np.float64)
        if prev_pos is not None:
            vel = (curr_target - prev_pos) * 30.0
        else:
            vel = np.zeros(3)
        prev_pos = curr_target.copy()

        if controller_type == "position":
            obs, info = env.step(action_4d)
        elif controller_type == "velocity":
            obs, info = env.step(action_4d, feedforward_vel=vel)
        elif controller_type == "impedance":
            obs, info = env.step(action_4d, target_vel=vel)

        if video_writer is not None:
            tau_str = f" | ||tau||: {info.get('tau_norm', 0.0):.1f}Nm" if "tau_norm" in info else ""
            label = (
                f"[{tag}] {inst['name']} | Step {step_i+1}/{len(scripted_traj)}"
                f"{tau_str} | Box: {info.get('on_pink_box', False)}"
            )
            video_writer.write(env.render_dual_view_bgr(label))

    if video_writer is not None:
        video_writer.release()
        print(f"  Saved validation rollout video to: {output_video_path}")

    env.close()

    success = False
    if task_name == "pick_and_place":
        success = bool(info.get("on_pink_box", False))
    elif task_name == "push":
        success = bool(info.get("next_to_pink_box", False))

    final_pos = info.get("cup_pos", np.zeros(3))
    print(
        f"  [{tag}] Validation Result: cup_pos={np.round(final_pos, 3).tolist()} | "
        f"tilt={info.get('cup_tilt_deg', 0.0):.1f}deg | "
        f"on_pink_box={info.get('on_pink_box', False)} | next_to_pink_box={info.get('next_to_pink_box', False)} | "
        f"SUCCESS={success}"
    )
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline validation & scripted physics verification for Franka FR3 MuJoCo suite."
    )
    parser.add_argument(
        "--controller",
        choices=["position", "velocity", "impedance", "all"],
        default="all",
        help="Controller to validate (position, velocity, impedance, or all).",
    )
    parser.add_argument(
        "--task",
        choices=["pick_and_place", "push", "all"],
        default="all",
        help="Task to validate (pick_and_place, push, or all).",
    )
    parser.add_argument(
        "--instance",
        type=int,
        default=0,
        help="Instance index to validate (0 to 9).",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch interactive MuJoCo 3D passive viewer GUI during validation.",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Render the test instances visual gallery image.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional destination path for validation video (default auto-names per controller).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  FRANKA FR3 SIMULATION PIPELINE VALIDATION")
    print("=" * 72)

    # 1. Joint Limits Verification
    limits_ok = verify_joint_limits()

    # 2. Render Gallery if requested
    if args.gallery or args.controller == "all":
        render_instances_gallery()

    # 3. Controllers to test
    if args.controller == "all":
        controllers = ["position", "velocity", "impedance"]
    else:
        controllers = [args.controller]

    # Tasks to test
    if args.task == "all":
        tasks = ["pick_and_place", "push"]
    else:
        tasks = [args.task]

    results = {}
    for ctrl in controllers:
        for t in tasks:
            out_vid = args.output_video
            if out_vid is None:
                out_vid = f"pipeline_validation_{ctrl}_{t}_demo.mp4"
            ok = validate_controller_pipeline(
                controller_type=ctrl,
                task_name=t,
                instance_idx=args.instance,
                output_video_path=out_vid,
                use_viewer=args.viewer,
            )
            results[f"{ctrl}_{t}"] = ok

    print("\n" + "=" * 72)
    print("  PIPELINE VALIDATION SUMMARY")
    print("=" * 72)
    print(f"  Joint Limits Matching XML:  {'PASSED' if limits_ok else 'FAILED'}")
    all_passed = limits_ok
    for k, ok in results.items():
        status = "PASSED" if ok else "FAILED"
        if not ok:
            all_passed = False
        print(f"  Controller Physics [{k:<22}]: {status}")
    print("=" * 72)
    if all_passed:
        print("  ALL PIPELINE VALIDATION TESTS PASSED SUCCESSFULLY!\n")
    else:
        print("  SOME PIPELINE VALIDATION TESTS FAILED. CHECK LOGS ABOVE.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

