#!/usr/bin/env python3
"""
Task-Space Differential Velocity Controller for Franka FR3 in MuJoCo.

Tracks predicted trajectories (from SmolVLA or reference instances) using task-space
differential velocity kinematics executed at the low level via direct motor torques:
- Damped least-squares (DLS) task-space differential kinematics:
    xdot_cmd = xdot_ref + Kp * (x_target - x_curr)
    w_cmd = Kr * rot_error(R_target, R_curr)
    qdot_des = J^T (J J^T + lambda^2 I)^(-1) [xdot_cmd; w_cmd] + (I - J^+ J) * K_null * (q_home - q)
- Direct joint torque motor commands on native <motor> actuators:
    tau_cmd = K_v * (qdot_des - qdot_meas) + tau_bias
    tau_bias = analytical gravity & Coriolis compensation (MuJoCo data.qfrc_bias)
- Actuators obey Franka FR3 torque limits: [-87, 87] Nm (joints 1-4), [-12, 12] Nm (joints 5-7).
"""

import argparse
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
    FrankaTaskSpaceController,
    build_test_scene_xml,
    default_wrist_up_rot_matrix,
    load_and_verify_checkpoint,
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


def build_velocity_scene_xml(
    fr3_xml_path: str = "franka_fr3/fr3.xml",
    calib_path: str = "camera_extrinsics.npz",
    cup_xy: Tuple[float, float] = (0.325, -0.055),
) -> str:
    """Build MuJoCo test scene with native <velocity> actuators replacing position actuators."""
    xml_str = build_test_scene_xml(
        fr3_xml_path=fr3_xml_path,
        calib_path=calib_path,
        cup_xy=cup_xy,
    )

    pos_actuators = """    <position class="fr3" name="fr3_joint1" joint="fr3_joint1" kp="4500" kv="450"/>
    <position class="fr3" name="fr3_joint2" joint="fr3_joint2" kp="4500" kv="450"/>
    <position class="fr3" name="fr3_joint3" joint="fr3_joint3" kp="3500" kv="350"/>
    <position class="fr3" name="fr3_joint4" joint="fr3_joint4" kp="3500" kv="350"/>
    <position class="fr3" name="fr3_joint5" joint="fr3_joint5" kp="2000" kv="200"/>
    <position class="fr3" name="fr3_joint6" joint="fr3_joint6" kp="2000" kv="200"/>
    <position class="fr3" name="fr3_joint7" joint="fr3_joint7" kp="2000" kv="200"/>"""

    # Direct torque <motor> actuators obeying Franka FR3 torque limits
    motor_actuators = """    <motor class="fr3" name="fr3_joint1" joint="fr3_joint1" ctrlrange="-87 87"/>
    <motor class="fr3" name="fr3_joint2" joint="fr3_joint2" ctrlrange="-87 87"/>
    <motor class="fr3" name="fr3_joint3" joint="fr3_joint3" ctrlrange="-87 87"/>
    <motor class="fr3" name="fr3_joint4" joint="fr3_joint4" ctrlrange="-87 87"/>
    <motor class="fr3" name="fr3_joint5" joint="fr3_joint5" ctrlrange="-12 12"/>
    <motor class="fr3" name="fr3_joint6" joint="fr3_joint6" ctrlrange="-12 12"/>
    <motor class="fr3" name="fr3_joint7" joint="fr3_joint7" ctrlrange="-12 12"/>"""

    return xml_str.replace(pos_actuators, motor_actuators)


class FrankaVelocityEnv:
    """
    MuJoCo simulation environment utilizing a task-space differential velocity controller
    executed at the low level via direct motor torque commands:
      tau = K_v * (qdot_des - qvel) + tau_bias
    where tau_bias is analytical gravity + Coriolis compensation from MuJoCo.
    """

    def __init__(
        self,
        fr3_xml_path: str = "franka_fr3/fr3.xml",
        calib_path: str = "camera_extrinsics.npz",
        control_hz: int = 30,
    ) -> None:
        self.fr3_xml_path = resolve_file(fr3_xml_path)
        self.calib_path = resolve_file(calib_path)
        self.control_hz = control_hz

        xml_str = build_velocity_scene_xml(self.fr3_xml_path, self.calib_path)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        self.data = mujoco.MjData(self.model)

        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fr3_hand")
        self.cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "green_cup")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pink_box")

        cup_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_freejoint")
        self.cup_qpos_adr = int(self.model.jnt_qposadr[cup_jnt_id])
        self.cup_qvel_adr = int(self.model.jnt_dofadr[cup_jnt_id])

        self.renderer_full = mujoco.Renderer(self.model, height=1080, width=1920)
        self.renderer_overview = mujoco.Renderer(self.model, height=720, width=960)

        self.physics_steps_per_ctrl = max(
            1, int(round(1.0 / (self.control_hz * self.model.opt.timestep)))
        )

        # Kinematic references and limits
        self.q_home = np.array([-0.044, 0.637, -0.030, -2.527, -1.930, 1.906, -2.328], dtype=np.float64)
        self.q_min = self.model.jnt_range[:7, 0].copy()
        self.q_max = self.model.jnt_range[:7, 1].copy()
        self.qdot_max = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float64)
        self.tau_max = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)

        self.R_target = default_wrist_up_rot_matrix(self.calib_path)
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.eye6 = (0.012 ** 2) * np.eye(6)
        self.eye7 = np.eye(7)

        self.task_space_controller = FrankaTaskSpaceController(self.model, self.data, self.calib_path)

        # Task-space kinematic velocity gains
        self.kp_pos = 16.0       # Position tracking feedback gain (1/s)
        self.kp_rot = 14.0       # Orientation tracking feedback gain (1/s)
        self.k_null = 2.5        # Nullspace posture pull gain (1/s)

        # Low-level direct joint torque velocity tracking gains (N*m*s/rad)
        self.kv_joint = np.array([140.0, 140.0, 120.0, 120.0, 45.0, 45.0, 45.0], dtype=np.float64)

        self.current_gripper_cmd = GRIPPER_OPEN
        self.reported_gripper_cmd = GRIPPER_OPEN
        self.close_counter = 0
        self.z_lift_offset = 0.0
        self.current_task_prompt = PROMPT_PICK_AND_PLACE

    def reset_instance(self, task_name: str, instance_idx: int = 0) -> Dict[str, Any]:
        """Reset the simulation to one of the standardized test instances."""
        inst = TEST_INSTANCES[task_name][instance_idx]
        cup_xy = inst["cup_xy"]
        self.current_task_prompt = inst["task_prompt"]

        mujoco.mj_resetData(self.model, self.data)

        init_grip = GRIPPER_OPEN if task_name == "pick_and_place" else GRIPPER_CLOSED
        if task_name == "pick_and_place":
            init_tcp = np.array([0.385, -0.105, 0.048], dtype=np.float64)
        else:
            cx, cy = inst["cup_xy"]
            init_tcp = np.array([cx, cy - 0.075, 0.048], dtype=np.float64)

        q_init = self.task_space_controller.solve_ik(
            init_tcp, q_init=self.task_space_controller.q_home, n_iter=80
        )
        self.data.qpos[:7] = q_init
        self.data.qvel[:] = 0.0

        self.current_gripper_cmd = init_grip
        self.reported_gripper_cmd = init_grip
        self.close_counter = 0 if init_grip >= 0.024 else 8
        self.z_lift_offset = 0.0

        # Set green cup position on table
        adr = self.cup_qpos_adr
        self.data.qpos[adr : adr + 7] = [float(cup_xy[0]), float(cup_xy[1]), 0.027, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7] = init_grip
        self.data.qpos[8] = init_grip
        self.data.ctrl[7] = init_grip
        self.data.ctrl[8] = init_grip

        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            self.data.ctrl[:7] = self.data.qfrc_bias[:7]
            mujoco.mj_step(self.model, self.data)

        return self.get_observation()

    def compute_joint_velocities(
        self,
        target_pos: np.ndarray,
        feedforward_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute desired joint velocities qdot via damped least-squares differential IK:
          qdot = J^T (J J^T + lambda^2 I)^(-1) [xdot; w] + (I - J^+ J) * K_null * (q_home - q)
        """
        curr_pos = self.data.site_xpos[self.tcp_id]
        pos_err = target_pos - curr_pos

        # Desired translational EEF velocity
        v_des = self.kp_pos * pos_err
        if feedforward_vel is not None:
            v_des += feedforward_vel
        v_des = np.clip(v_des, -0.45, 0.45)

        # Desired rotational EEF velocity (wrist-up pitch)
        R_curr = self.data.xmat[self.hand_id].reshape(3, 3)
        R_err = self.R_target @ R_curr.T
        rot_vec, _ = cv2.Rodrigues(R_err)
        w_des = np.clip(self.kp_rot * rot_vec.ravel(), -1.8, 1.8)

        # Compute Jacobians
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp_id)
        J = np.vstack([self.jacp[:, :7], 0.8 * self.jacr[:, :7]])
        dx_cmd = np.concatenate([v_des, 0.8 * w_des])

        # Damped least squares
        JJT = J @ J.T + self.eye6
        dq_task = J.T @ np.linalg.solve(JJT, dx_cmd)

        # Nullspace posture regularization
        N = self.eye7 - J.T @ np.linalg.solve(JJT, J)
        dq_null = N @ (self.k_null * (self.q_home - self.data.qpos[:7]))

        dq_cmd = dq_task + dq_null
        return np.clip(dq_cmd, -self.qdot_max, self.qdot_max)

    def compute_joint_torques(
        self,
        target_pos: np.ndarray,
        feedforward_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute low-level direct joint torques tracking task-space differential velocity:
          tau_cmd = K_v * (qdot_des - qdot_meas) + tau_bias
        where tau_bias = data.qfrc_bias[:7] provides analytical gravity + Coriolis compensation.
        """
        dq_des = self.compute_joint_velocities(target_pos, feedforward_vel)
        tau_vel = self.kv_joint * (dq_des - self.data.qvel[:7])
        tau_bias = self.data.qfrc_bias[:7]
        tau_cmd = tau_vel + tau_bias
        return np.clip(tau_cmd, -self.tau_max, self.tau_max)

    def step(
        self,
        target_pos_4d: np.ndarray,
        feedforward_vel: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Advance the simulation by one 30 Hz control cycle using the velocity controller:
        - target_pos_4d: [target_tcp_x, target_tcp_y, target_tcp_z, gripper_cmd]
        - Direct joint torque commands applied to <motor> actuators at 1000 Hz.
        """
        grip_raw = float(target_pos_4d[3])
        if grip_raw < 0.024:
            self.close_counter += 1
            self.reported_gripper_cmd = GRIPPER_CLOSED
        else:
            self.close_counter = 0
            self.reported_gripper_cmd = GRIPPER_OPEN

        phys_grip = GRIPPER_CLOSED if self.close_counter >= 8 else GRIPPER_OPEN
        self.current_gripper_cmd = phys_grip

        f_open = 0.5 * (self.data.qpos[7] + self.data.qpos[8])
        cup_grasped = (phys_grip == GRIPPER_CLOSED) and (f_open > 0.010)

        # Wall clearance offset over pink box
        if cup_grasped and float(target_pos_4d[1]) > -0.035 and float(target_pos_4d[0]) > 0.275:
            target_z_off = 0.022
        else:
            target_z_off = 0.0
        self.z_lift_offset = 0.80 * self.z_lift_offset + 0.20 * target_z_off

        target_pos = np.array(target_pos_4d[:3], dtype=np.float64)
        target_pos[2] += self.z_lift_offset
        target_pos[0] = float(np.clip(target_pos[0], 0.14, 0.50))
        target_pos[1] = float(np.clip(target_pos[1], -0.22, 0.08))
        target_pos[2] = float(np.clip(target_pos[2], 0.036, 0.24))

        adr = self.cup_qpos_adr
        qvel_adr = self.cup_qvel_adr

        start_pos = self.data.site_xpos[self.tcp_id].copy()
        n_sub = self.physics_steps_per_ctrl
        last_tau_cmd = np.zeros(7)

        # Run internal physics steps at 1000 Hz, applying direct joint torques
        for sub_i in range(n_sub):
            alpha = float(sub_i + 1) / float(n_sub)
            interp_target = (1.0 - alpha) * start_pos + alpha * target_pos
            tau_cmd = self.compute_joint_torques(interp_target, feedforward_vel)
            last_tau_cmd = tau_cmd

            self.data.ctrl[:7] = tau_cmd
            self.data.ctrl[7] = phys_grip
            self.data.ctrl[8] = phys_grip

            mujoco.mj_step(self.model, self.data)
            if cup_grasped:
                self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[qvel_adr + 3 : qvel_adr + 6] = 0.0

        cup_pos = self.data.xpos[self.cup_body_id].copy()
        box_pos = self.data.xpos[self.box_body_id].copy()
        cup_mat = self.data.xmat[self.cup_body_id].reshape(3, 3)
        cup_tilt_deg = float(np.degrees(np.arccos(np.clip(cup_mat[2, 2], -1.0, 1.0))))
        cup_upright = cup_tilt_deg < 25.0

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

        tracking_error = float(np.linalg.norm(target_pos - self.data.site_xpos[self.tcp_id]))
        info = {
            "cup_pos": cup_pos,
            "cup_tilt_deg": cup_tilt_deg,
            "tcp_pos": self.data.site_xpos[self.tcp_id].copy(),
            "tracking_error_m": tracking_error,
            "tau_norm": float(np.linalg.norm(last_tau_cmd)),
            "on_pink_box": on_pink_box,
            "next_to_pink_box": next_to_pink_box,
        }
        return self.get_observation(), info

    def get_state_4d(self) -> np.ndarray:
        tcp_pos = self.data.site_xpos[self.tcp_id].astype(np.float32)
        reported_z = float(tcp_pos[2]) - float(self.z_lift_offset)
        return np.array([tcp_pos[0], tcp_pos[1], reported_z, float(self.reported_gripper_cmd)], dtype=np.float32)

    def render_smolvla_input(self) -> np.ndarray:
        self.renderer_full.update_scene(self.data, camera="real_cam")
        full_rgb = self.renderer_full.render()
        cropped = full_rgb[:, CROP_X_START:]
        return cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    def render_overview(self, camera_name: str = "overview_cam") -> np.ndarray:
        self.renderer_overview.update_scene(self.data, camera=camera_name)
        return self.renderer_overview.render().copy()

    def render_dual_view_bgr(self, label_text: str = "") -> np.ndarray:
        left_bgr = cv2.cvtColor(self.render_smolvla_input(), cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(
            cv2.resize(self.render_overview(), (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA),
            cv2.COLOR_RGB2BGR,
        )
        cv2.putText(left_bgr, "In-Person Camera (Velocity Control)", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(right_bgr, "3rd-Person Overview", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        if label_text:
            cv2.putText(left_bgr, label_text, (14, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 200), 2)
        return np.hstack([left_bgr, right_bgr])

    def get_observation(self) -> Dict[str, Any]:
        return {
            "observation.images.camera1": self.render_smolvla_input(),
            "observation.state": self.get_state_4d(),
            "task": self.current_task_prompt,
        }

    def close(self) -> None:
        self.renderer_full.close()
        self.renderer_overview.close()


def run_velocity_evaluation(
    checkpoint_path: str = "last",
    task_name: Optional[str] = "pick_and_place",
    prompt_input: Optional[str] = None,
    instance_filter: Optional[Any] = None,
    output_video_path: str = "velocity_controller_rollout.mp4",
    max_steps: int = 165,
    device: str = "cuda",
    use_viewer: bool = False,
) -> List[Dict[str, Any]]:
    """Run closed-loop evaluation across test instances using the velocity controller."""
    # Strictly load and verify SmolVLA checkpoint (throws FileNotFoundError if missing)
    resolved_ckpt = load_and_verify_checkpoint(checkpoint_path)
    print(f"Loading SmolVLA checkpoint from {resolved_ckpt} ...", flush=True)

    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

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

    env = FrankaVelocityEnv()

    # Interactive MuJoCo GUI viewer if requested and display available
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
                print("DISPLAY environment variable not set; defaulting to headless video.", flush=True)
        except Exception as e:
            print(f"Interactive viewer GUI not available ({e}). Defaulting to video.", flush=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (TARGET_WIDTH * 2, TARGET_HEIGHT))

    tasks_to_run = (
        ["pick_and_place", "push"]
        if (task_name in [None, "all"] and instance_filter is None)
        else [task_name] if task_name in ["pick_and_place", "push"]
        else ["pick_and_place", "push"]
    )

    all_results: List[Dict[str, Any]] = []

    for cur_task in tasks_to_run:
        active_prompt = PROMPT_PUSH if cur_task == "push" else PROMPT_PICK_AND_PLACE
        if prompt_input:
            active_prompt, _ = resolve_prompt(prompt_input, default_task=cur_task)

        for idx, inst in enumerate(TEST_INSTANCES[cur_task]):
            # Check instance filter
            if instance_filter is not None:
                inst_str = str(instance_filter).strip()
                if inst_str.isdigit() and int(inst_str) != idx:
                    continue
                elif not inst_str.isdigit() and inst_str not in [inst["name"], cur_task, "all"]:
                    continue

            obs = env.reset_instance(cur_task, idx)
            policy.reset()
            env.current_task_prompt = active_prompt
            obs["task"] = active_prompt

            print(f"Running {inst['name']} ({inst['description']}) [Velocity Controller]...", flush=True)
            info: Dict[str, Any] = {}
            prev_pos = None
            tracking_errors = []
            tau_norms = []

            for step_i in range(max_steps):
                if viewer is not None:
                    if not viewer.is_running():
                        print("Viewer window closed by user. Terminating rollout.", flush=True)
                        break
                    viewer.sync()

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

                curr_target = np.array(action_4d[:3], dtype=np.float64)
                if prev_pos is not None:
                    feedforward_vel = (curr_target - prev_pos) * 30.0
                else:
                    feedforward_vel = np.zeros(3)
                prev_pos = curr_target.copy()

                obs, info = env.step(action_4d, feedforward_vel=feedforward_vel)
                tracking_errors.append(info.get("tracking_error_m", 0.0))
                tau_norms.append(info.get("tau_norm", 0.0))

                label = (
                    f"{inst['name']} | Step {step_i+1} | Err: {info['tracking_error_m']*100:.1f}cm | "
                    f"||tau||: {info.get('tau_norm', 0.0):.1f}Nm | Box: {info['on_pink_box']}"
                )
                writer.write(env.render_dual_view_bgr(label))

            success = bool(info.get("on_pink_box", False)) if cur_task == "pick_and_place" else bool(info.get("next_to_pink_box", False))
            cup_pos_list = [round(x, 3) for x in info.get("cup_pos", np.zeros(3)).tolist()]
            mean_err_cm = float(np.mean(tracking_errors)) * 100.0 if tracking_errors else 0.0
            mean_tau = float(np.mean(tau_norms)) if tau_norms else 0.0

            result_entry = {
                "name": inst["name"],
                "task": cur_task,
                "success": success,
                "cup_pos": cup_pos_list,
                "cup_tilt_deg": round(info.get("cup_tilt_deg", 0.0), 1),
                "on_pink_box": bool(info.get("on_pink_box", False)),
                "next_to_pink_box": bool(info.get("next_to_pink_box", False)),
                "mean_tracking_err_cm": round(mean_err_cm, 2),
                "mean_tau_norm": round(mean_tau, 2),
            }
            all_results.append(result_entry)

            print(
                f"  Result: cup_pos={cup_pos_list} | tilt={info.get('cup_tilt_deg', 0.0):.1f}deg | "
                f"on_pink_box={info.get('on_pink_box', False)} | next_to_pink_box={info.get('next_to_pink_box', False)} | "
                f"err={mean_err_cm:.1f}cm | ||tau||={mean_tau:.1f}Nm | success={'PASS' if success else 'FAIL'}",
                flush=True,
            )

    writer.release()
    env.close()
    print(f"Saved velocity rollout video to: {output_video_path}", flush=True)

    # Print summary if multiple instances were run
    if len(all_results) > 1:
        pick_results = [r for r in all_results if r["task"] == "pick_and_place"]
        push_results = [r for r in all_results if r["task"] == "push"]
        print("\n" + "=" * 60)
        print("  VELOCITY CONTROLLER EVALUATION SUMMARY")
        print("=" * 60)
        if pick_results:
            n_p = sum(1 for r in pick_results if r["success"])
            print(f"  Pick-and-Place: {n_p}/{len(pick_results)} ({n_p/len(pick_results)*100:.1f}%)")
        if push_results:
            n_u = sum(1 for r in push_results if r["success"])
            print(f"  Push:           {n_u}/{len(push_results)} ({n_u/len(push_results)*100:.1f}%)")
        total_pass = sum(1 for r in all_results if r["success"])
        print(f"  Total:          {total_pass}/{len(all_results)} ({total_pass/len(all_results)*100:.1f}%)")
        print("=" * 60 + "\n", flush=True)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run task-space differential velocity controller in MuJoCo (direct torque motor execution)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Path or identifier of fine-tuned SmolVLA checkpoint (default: 'last'). Throws an error if not found.",
    )
    parser.add_argument(
        "--task",
        choices=["pick_and_place", "push", "all"],
        default="all",
        help="Task to execute (pick_and_place, push, or all; default: all).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Natural language instruction for SmolVLA (e.g. 'Pick up the green cup and place it on the pink box').",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Optional test instance index (0 to 9), name (e.g. 'pick_01_nominal_center'), or None to evaluate all instances.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="velocity_controller_rollout.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=165,
        help="Maximum steps for policy evaluation.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch interactive MuJoCo 3D passive viewer GUI.",
    )
    import torch
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_dev,
        help="Inference device (cuda or cpu).",
    )
    args = parser.parse_args()

    run_velocity_evaluation(
        checkpoint_path=args.checkpoint,
        task_name=args.task,
        prompt_input=args.prompt,
        instance_filter=args.instance,
        output_video_path=args.output_video,
        max_steps=args.max_steps,
        device=args.device,
        use_viewer=args.viewer,
    )


if __name__ == "__main__":
    main()

