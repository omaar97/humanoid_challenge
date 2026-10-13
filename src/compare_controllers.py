#!/usr/bin/env python3
"""
Comprehensive Comparison Runner for SmolVLA Control Architectures in MuJoCo.

Compares all 3 control methods across the standardized tabletop test scenarios:
1. Position Control: Task-space Nullspace Inverse Kinematics position servoing (mujoco_main_test_env.py)
2. Velocity Control: Differential kinematics task-space velocity control via direct motor torques (mujoco_velocity_controller_test.py)
3. Impedance Control: Pure End-Effector Cartesian impedance control via direct motor torques (mujoco_impedance_controller_test.py)

Outputs:
- Live progress and rollout results in console
- Formatted side-by-side comparison matrix printed to stdout
- Markdown summary report: controller_comparison_results.md
- Machine-readable JSON summary: controller_comparison_results.json
"""

import argparse
import gc
import json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mujoco_main_test_env import (
    PROMPT_PICK_AND_PLACE,
    PROMPT_PUSH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TEST_INSTANCES,
    FrankaCupMuJoCoEnv,
    load_and_verify_checkpoint,
)
from mujoco_velocity_controller_test import FrankaVelocityEnv
from mujoco_impedance_controller_test import FrankaImpedanceEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def run_single_rollout(
    controller_type: str,
    env: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    task_name: str,
    instance_idx: int,
    max_steps: int = 165,
    writer: Optional[cv2.VideoWriter] = None,
    record_frames: bool = False,
) -> Dict[str, Any]:
    """Execute a single episode for the given controller and instance."""
    inst = TEST_INSTANCES[task_name][instance_idx]
    active_prompt = PROMPT_PUSH if task_name == "push" else PROMPT_PICK_AND_PLACE

    obs = env.reset_instance(task_name, instance_idx)
    policy.reset()
    env.current_task_prompt = active_prompt
    obs["task"] = active_prompt

    info: Dict[str, Any] = {}
    prev_pos: Optional[np.ndarray] = None
    tracking_errors: List[float] = []
    tau_norms: List[float] = []
    frames_buffer: List[np.ndarray] = []

    for step_i in range(max_steps):

        img_t = torch.from_numpy(obs["observation.images.camera1"]).permute(2, 0, 1).float().div(255.0)
        state_t = torch.from_numpy(obs["observation.state"]).float()
        raw_batch = {
            "observation.images.camera1": img_t,
            "observation.state": state_t,
            "task": active_prompt,
        }
        batch = preprocessor(raw_batch)
        with torch.inference_mode():
            act_t = policy.select_action(batch)
        act_unnorm = postprocessor(act_t)
        action_4d = act_unnorm.squeeze(0).cpu().numpy()

        if controller_type == "position":
            obs, info = env.step(action_4d)
        elif controller_type == "velocity":
            curr_target = np.array(action_4d[:3], dtype=np.float64)
            if prev_pos is not None:
                ff_vel = (curr_target - prev_pos) * 30.0
            else:
                ff_vel = np.zeros(3)
            prev_pos = curr_target.copy()
            obs, info = env.step(action_4d, feedforward_vel=ff_vel)
            tracking_errors.append(info.get("tracking_error_m", 0.0))
            tau_norms.append(info.get("tau_norm", 0.0))
        elif controller_type == "impedance":
            curr_target = np.array(action_4d[:3], dtype=np.float64)
            if prev_pos is not None:
                t_vel = (curr_target - prev_pos) * 30.0
            else:
                t_vel = np.zeros(3)
            prev_pos = curr_target.copy()
            obs, info = env.step(action_4d, target_vel=t_vel)
            tracking_errors.append(info.get("tracking_error_m", 0.0))
            tau_norms.append(info.get("tau_norm", 0.0))

        if record_frames:
            overview_rgb = env.render_overview("overview_cam")
            overview_bgr = cv2.cvtColor(overview_rgb, cv2.COLOR_RGB2BGR)
            frames_buffer.append(overview_bgr)
        elif writer is not None:
            label = f"{controller_type.upper()} | {inst['name']} | step {step_i+1}/{max_steps}"
            writer.write(env.render_dual_view_bgr(label))

    success = bool(info.get("on_pink_box", False)) if task_name == "pick_and_place" else bool(info.get("next_to_pink_box", False))
    cup_pos_list = [round(float(x), 3) for x in info.get("cup_pos", np.zeros(3)).tolist()]
    mean_err_cm = float(np.mean(tracking_errors)) * 100.0 if tracking_errors else 0.0
    mean_tau = float(np.mean(tau_norms)) if tau_norms else 0.0

    return {
        "controller": controller_type,
        "name": inst["name"],
        "task": task_name,
        "instance_idx": instance_idx,
        "success": success,
        "cup_pos": cup_pos_list,
        "cup_tilt_deg": round(float(info.get("cup_tilt_deg", 0.0)), 1),
        "on_pink_box": bool(info.get("on_pink_box", False)),
        "next_to_pink_box": bool(info.get("next_to_pink_box", False)),
        "mean_tracking_err_cm": round(mean_err_cm, 2),
        "mean_tau_norm": round(mean_tau, 2),
        "frames": frames_buffer if record_frames else [],
    }


def compare_controllers(
    checkpoint_path: str = "last",
    tasks: Optional[List[str]] = None,
    controllers_to_test: Optional[List[str]] = None,
    instance_filter: Optional[str] = None,
    max_steps: int = 165,
    device: str = "cuda",
    output_markdown_path: str = "controller_comparison_results.md",
    output_json_path: str = "controller_comparison_results.json",
    output_video_path: Optional[str] = None,
    split_screen: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation across the chosen controllers and compile side-by-side comparison metrics."""
    if tasks is None:
        tasks = ["pick_and_place", "push"]
    if controllers_to_test is None:
        controllers_to_test = ["position", "velocity", "impedance"]

    resolved_ckpt = load_and_verify_checkpoint(checkpoint_path)
    print(f"\nLoading SmolVLA checkpoint from {resolved_ckpt} on {device} ...", flush=True)

    tok_dir = os.path.join(resolved_ckpt, "tokenizer")
    pre_overrides: Dict[str, Any] = {"device_processor": {"device": device}}
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

    # Initialize environments for each requested controller
    envs: Dict[str, Any] = {}
    if "position" in controllers_to_test:
        envs["position"] = FrankaCupMuJoCoEnv()
    if "velocity" in controllers_to_test:
        envs["velocity"] = FrankaVelocityEnv()
    if "impedance" in controllers_to_test:
        envs["impedance"] = FrankaImpedanceEnv()

    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        if split_screen and len(controllers_to_test) > 1:
            frame_w = TARGET_WIDTH * len(controllers_to_test)
            frame_h = TARGET_HEIGHT
        else:
            frame_w = TARGET_WIDTH * 2
            frame_h = TARGET_HEIGHT
        writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_w, frame_h))
        mode_str = f"Split-screen ({frame_w}x{frame_h})" if split_screen else f"Dual-view sequential ({frame_w}x{frame_h})"
        print(f"Recording comparison video [{mode_str}] to: {output_video_path}", flush=True)

    print("\n" + "=" * 80)
    print("  STARTING CONTROLLER COMPARISON BENCHMARK")
    print(f"  Controllers: {[c.capitalize() for c in controllers_to_test]}")
    print(f"  Tasks:       {tasks}")
    print("=" * 80 + "\n", flush=True)

    # Results structured as: results[ctrl_type][task_name][instance_name]
    raw_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in controllers_to_test}

    for t_name in tasks:
        print(f"\n--- EVALUATING TASK: {t_name.upper()} ---", flush=True)
        for idx, inst in enumerate(TEST_INSTANCES[t_name]):
            if instance_filter is not None:
                filt = str(instance_filter).strip()
                if filt.isdigit() and int(filt) != idx:
                    continue
                elif not filt.isdigit() and filt not in [inst["name"], t_name, "all"]:
                    continue

            print(f"\nScenario [{idx+1}/10]: {inst['name']} ({inst['description']})")
            record_split = bool(split_screen and output_video_path)
            scenario_rollouts: Dict[str, Dict[str, Any]] = {}
            for c_name in controllers_to_test:
                env = envs[c_name]
                res = run_single_rollout(
                    controller_type=c_name,
                    env=env,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_name=t_name,
                    instance_idx=idx,
                    max_steps=max_steps,
                    writer=writer if not record_split else None,
                    record_frames=record_split,
                )
                raw_results[c_name].append(res)
                scenario_rollouts[c_name] = res
                status_str = "PASS" if res["success"] else "FAIL"
                extra_metric = f" | err={res['mean_tracking_err_cm']}cm | ||tau||={res['mean_tau_norm']}Nm" if c_name != "position" else ""
                print(f"  -> {c_name.capitalize():<11}: {status_str} (tilt={res['cup_tilt_deg']}°, pos={res['cup_pos']}{extra_metric})", flush=True)

            if record_split and writer is not None:
                min_len = min(len(scenario_rollouts[c]["frames"]) for c in controllers_to_test)
                for step_idx in range(min_len):
                    panels = []
                    for c_name in controllers_to_test:
                        raw_frame = scenario_rollouts[c_name]["frames"][step_idx].copy()
                        panel = cv2.resize(raw_frame, (TARGET_WIDTH, TARGET_HEIGHT))
                        c_title = f"{c_name.upper()} | {inst['name']}"
                        c_res = scenario_rollouts[c_name]
                        # Top header bar
                        cv2.rectangle(panel, (0, 0), (TARGET_WIDTH, 42), (20, 20, 20), -1)
                        color = (0, 255, 255) if c_name == "position" else (0, 255, 100) if c_name == "velocity" else (255, 150, 50)
                        cv2.putText(panel, c_title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
                        # Bottom info bar
                        cv2.rectangle(panel, (0, TARGET_HEIGHT - 38), (TARGET_WIDTH, TARGET_HEIGHT), (20, 20, 20), -1)
                        if c_name != "position":
                            metrics_str = f"Err: {c_res['mean_tracking_err_cm']:.1f}cm | ||tau||: {c_res['mean_tau_norm']:.1f}Nm"
                        else:
                            metrics_str = "Kinematic Nullspace IK Servo"
                        cv2.putText(panel, metrics_str, (14, TARGET_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
                        # Status badge near end
                        if step_idx >= min_len - 30:
                            is_p = c_res["success"]
                            tag = "PASS" if is_p else "FAIL"
                            tcolor = (0, 220, 0) if is_p else (0, 0, 240)
                            cv2.putText(panel, tag, (TARGET_WIDTH - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcolor, 2, cv2.LINE_AA)
                        panels.append(panel)
                    combined_split = np.hstack(panels)
                    writer.write(combined_split)
                # Free frames immediately to avoid high RAM usage
                for c_name in controllers_to_test:
                    scenario_rollouts[c_name]["frames"].clear()
                gc.collect()

    if writer is not None:
        writer.release()
        print(f"\nSaved comparison rollout video to: {output_video_path}", flush=True)

    for env in envs.values():
        env.close()

    # -------------------------------------------------------------------------
    # COMPILE SUMMARY & COMPARISON TABLES
    # -------------------------------------------------------------------------
    matrix: Dict[str, Any] = {"scenarios": [], "summary": {}}

    # Collect all scenario names in order
    all_scenarios: List[Tuple[str, str]] = []
    for t_name in tasks:
        for idx, inst in enumerate(TEST_INSTANCES[t_name]):
            if instance_filter is not None:
                filt = str(instance_filter).strip()
                if filt.isdigit() and int(filt) != idx:
                    continue
                elif not filt.isdigit() and filt not in [inst["name"], t_name, "all"]:
                    continue
            all_scenarios.append((t_name, inst["name"]))

    scenario_rows: List[Dict[str, Any]] = []
    for t_name, sc_name in all_scenarios:
        row: Dict[str, Any] = {"task": t_name, "scenario": sc_name, "controllers": {}}
        for c_name in controllers_to_test:
            match = next((r for r in raw_results[c_name] if r["name"] == sc_name and r["task"] == t_name), None)
            if match:
                row["controllers"][c_name] = {
                    "success": match["success"],
                    "tilt": match["cup_tilt_deg"],
                    "pos": match["cup_pos"],
                    "err_cm": match["mean_tracking_err_cm"],
                    "tau_norm": match["mean_tau_norm"],
                }
        scenario_rows.append(row)
    matrix["scenarios"] = scenario_rows

    # Compile Summary Per Controller
    summary_by_ctrl: Dict[str, Any] = {}
    for c_name in controllers_to_test:
        c_res = raw_results[c_name]
        pick_res = [r for r in c_res if r["task"] == "pick_and_place"]
        push_res = [r for r in c_res if r["task"] == "push"]

        pick_pass = sum(1 for r in pick_res if r["success"])
        push_pass = sum(1 for r in push_res if r["success"])
        total_pass = sum(1 for r in c_res if r["success"])

        mean_err = float(np.mean([r["mean_tracking_err_cm"] for r in c_res])) if c_name != "position" and c_res else None
        mean_tau = float(np.mean([r["mean_tau_norm"] for r in c_res])) if c_name != "position" and c_res else None

        summary_by_ctrl[c_name] = {
            "pick_success": f"{pick_pass}/{len(pick_res)}" if pick_res else "N/A",
            "pick_rate": round(pick_pass / len(pick_res) * 100.0, 1) if pick_res else 0.0,
            "push_success": f"{push_pass}/{len(push_res)}" if push_res else "N/A",
            "push_rate": round(push_pass / len(push_res) * 100.0, 1) if push_res else 0.0,
            "total_success": f"{total_pass}/{len(c_res)}" if c_res else "N/A",
            "total_rate": round(total_pass / len(c_res) * 100.0, 1) if c_res else 0.0,
            "mean_tracking_err_cm": mean_err,
            "mean_tau_norm": mean_tau,
        }
    matrix["summary"] = summary_by_ctrl

    # -------------------------------------------------------------------------
    # PRINT FORMATTED COMPARISON MATRIX TO STDOUT
    # -------------------------------------------------------------------------
    header_ctrls = "".join([f"| {c.capitalize() + ' Control':^18} " for c in controllers_to_test])
    line_div = "-" * (36 + 21 * len(controllers_to_test))

    print("\n" + "=" * len(line_div))
    print("                    SMOLVLA CONTROLLER COMPARISON MATRIX")
    print("=" * len(line_div))
    print(f"{'Scenario':<26} | {'Task':<5} " + header_ctrls)
    print(line_div)

    for row in scenario_rows:
        task_short = "Pick" if row["task"] == "pick_and_place" else "Push"
        cols = []
        for c in controllers_to_test:
            c_info = row["controllers"].get(c, {})
            status = "PASS" if c_info.get("success", False) else "FAIL"
            tilt = c_info.get("tilt", 0.0)
            cell = f"{status} ({tilt:.1f}°)"
            cols.append(f"| {cell:^18} ")
        print(f"{row['scenario']:<26} | {task_short:<5} " + "".join(cols))

    print(line_div)
    print("SUMMARY PERFORMANCE:")
    print(line_div)

    # Pick success row
    pick_cells = []
    for c in controllers_to_test:
        s = summary_by_ctrl[c]
        pick_cells.append(f"| {s['pick_success'] + ' (' + str(s['pick_rate']) + '%)':^18} ")
    print(f"{'Pick & Place Success Rate':<34} " + "".join(pick_cells))

    # Push success row
    push_cells = []
    for c in controllers_to_test:
        s = summary_by_ctrl[c]
        push_cells.append(f"| {s['push_success'] + ' (' + str(s['push_rate']) + '%)':^18} ")
    print(f"{'Push Success Rate':<34} " + "".join(push_cells))

    # Overall success row
    total_cells = []
    for c in controllers_to_test:
        s = summary_by_ctrl[c]
        total_cells.append(f"| {s['total_success'] + ' (' + str(s['total_rate']) + '%)':^18} ")
    print(f"{'Overall Success Rate':<34} " + "".join(total_cells))

    # Tracking error row
    err_cells = []
    for c in controllers_to_test:
        err = summary_by_ctrl[c]["mean_tracking_err_cm"]
        cell = f"{err:.1f} cm" if err is not None else "N/A (Pos Servo)"
        err_cells.append(f"| {cell:^18} ")
    print(f"{'Mean Tracking Error':<34} " + "".join(err_cells))

    # Torque effort row
    tau_cells = []
    for c in controllers_to_test:
        tau = summary_by_ctrl[c]["mean_tau_norm"]
        cell = f"{tau:.1f} Nm" if tau is not None else "N/A (Kinematic)"
        tau_cells.append(f"| {cell:^18} ")
    print(f"{'Mean Torque Effort ||tau||':<34} " + "".join(tau_cells))
    print("=" * len(line_div) + "\n", flush=True)

    # -------------------------------------------------------------------------
    # WRITE MARKDOWN REPORT
    # -------------------------------------------------------------------------
    md_content = []
    md_content.append("# SmolVLA Controller Comparison Report\n")
    md_content.append(f"Evaluated on fine-tuned checkpoint: `{resolved_ckpt}`\n")
    md_content.append("## 1. Summary Performance Metrics\n")
    md_content.append("| Metric | " + " | ".join([f"{c.capitalize()} Control" for c in controllers_to_test]) + " |")
    md_content.append("| :--- | " + " | ".join([":---:" for _ in controllers_to_test]) + " |")

    md_content.append("| **Pick & Place Success** | " + " | ".join([f"**{summary_by_ctrl[c]['pick_success']} ({summary_by_ctrl[c]['pick_rate']}%)**" for c in controllers_to_test]) + " |")
    md_content.append("| **Push Success** | " + " | ".join([f"**{summary_by_ctrl[c]['push_success']} ({summary_by_ctrl[c]['push_rate']}%)**" for c in controllers_to_test]) + " |")
    md_content.append("| **Overall Success** | " + " | ".join([f"**{summary_by_ctrl[c]['total_success']} ({summary_by_ctrl[c]['total_rate']}%)**" for c in controllers_to_test]) + " |")
    md_content.append("| **Mean Tracking Error** | " + " | ".join([f"{summary_by_ctrl[c]['mean_tracking_err_cm']:.1f} cm" if summary_by_ctrl[c]['mean_tracking_err_cm'] is not None else "N/A (Kinematic)" for c in controllers_to_test]) + " |")
    md_content.append("| **Mean Torque Effort ||tau||** | " + " | ".join([f"{summary_by_ctrl[c]['mean_tau_norm']:.1f} Nm" if summary_by_ctrl[c]['mean_tau_norm'] is not None else "N/A" for c in controllers_to_test]) + " |\n")

    md_content.append("## 2. Per-Scenario Comparison Table\n")
    md_content.append("| Scenario | Task | " + " | ".join([f"{c.capitalize()} Control" for c in controllers_to_test]) + " |")
    md_content.append("| :--- | :--- | " + " | ".join([":---:" for _ in controllers_to_test]) + " |")

    for row in scenario_rows:
        task_label = "Pick & Place" if row["task"] == "pick_and_place" else "Push"
        cols = []
        for c in controllers_to_test:
            c_info = row["controllers"].get(c, {})
            pass_str = "**PASS**" if c_info.get("success", False) else "FAIL"
            cols.append(f"{pass_str} (tilt={c_info.get('tilt', 0.0):.1f}°)")
        md_content.append(f"| `{row['scenario']}` | {task_label} | " + " | ".join(cols) + " |")

    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")
    print(f"Saved Markdown comparison report to: {output_markdown_path}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    print(f"Saved JSON comparison data to: {output_json_path}")

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run side-by-side comparison across Position, Velocity, and Impedance controllers for SmolVLA."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Path or identifier of fine-tuned SmolVLA checkpoint (default: 'last').",
    )
    parser.add_argument(
        "--task",
        choices=["pick_and_place", "push", "all"],
        default="all",
        help="Task to evaluate ('pick_and_place', 'push', or 'all'; default: 'all').",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="position,velocity,impedance",
        help="Comma-separated list of controllers to benchmark (default: 'position,velocity,impedance').",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Optional single instance name or index (e.g. 'pick_01_nominal_center' or '0'). Evaluates all if omitted.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=165,
        help="Maximum control steps per episode.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path to record comparison rollouts to an MP4 video (default: None - video recording disabled).",
    )
    parser.add_argument(
        "--split-screen",
        action="store_true",
        help="Generate a synchronized side-by-side split-screen video comparing all controllers simultaneously (requires --output-video).",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="controller_comparison_results.md",
        help="Path for generated Markdown comparison table.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="controller_comparison_results.json",
        help="Path for generated JSON comparison data.",
    )
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help="Inference device (cuda or cpu).",
    )
    args = parser.parse_args()

    task_list = ["pick_and_place", "push"] if args.task == "all" else [args.task]
    ctrl_list = [c.strip().lower() for c in args.controllers.split(",") if c.strip()]
    video_path = args.output_video

    compare_controllers(
        checkpoint_path=args.checkpoint,
        tasks=task_list,
        controllers_to_test=ctrl_list,
        instance_filter=args.instance,
        max_steps=args.max_steps,
        device=args.device,
        output_markdown_path=args.output_markdown,
        output_json_path=args.output_json,
        output_video_path=video_path,
        split_screen=args.split_screen,
    )


if __name__ == "__main__":
    main()

