# Human-to-Franka FR3 Imitation Learning with SmolVLA & MuJoCo

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SmolVLA%20Model-FFD21E)](https://huggingface.co/omaar97/smolvla_humanoid_challenge)

This repository implements an end-to-end pipeline that converts **raw monocular RGB videos of a human hand** performing tabletop manipulation tasks (`raw_pick_and_place.mp4` and `raw_push.mp4`) into a **robot demonstration dataset** (`lerobot_dataset` v3.0), fine-tunes a **SmolVLA** Vision-Language-Action policy (`lerobot/smolvla_base`), and evaluates the learned policy closed-loop inside a **MuJoCo simulation environment** using the 7-DOF **Franka Emika FR3** arm with different controllers: **position-based inverse-kinematics (default/main), velocity-based differential inverse kinematics, and impedance controllers**.

> [!WARNING]
> **Disclaimer**
>
> This code was written with the help of LLM agents. Many parts of the code are ad hoc to the specific collected dataset and have not been modified to generalize to other datasets. This README outlines the limitations of the scripts.

> The Franka FR3 model in [`franka_fr3/`](franka_fr3/) is taken from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3).

### Hosted Assets on Hugging Face (`omaar97`)
To keep this GitHub repository lightweight, all large binary assets (dataset, fine-tuned weights, and videos) are hosted on Hugging Face:
- **LeRobot Dataset & Videos (`omaar97/humanoid_challenge_dataset`)**: [https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset)
  - [20-Scenario SmolVLA Evaluation Video (`smolvla_eval_rollouts.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/smolvla_eval_rollouts.mp4)
  - [MuJoCo Franka Overlay — Pick & Place (`franka_overlay_pick_and_place.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/franka_overlay_pick_and_place.mp4)
  - [MuJoCo Franka Overlay — Push (`franka_overlay_push.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/franka_overlay_push.mp4)
  - [SAM 2 Hand-Inpainted Clean Video — Pick & Place (`masked_clean_pick_and_place.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/masked_clean_pick_and_place.mp4)
  - [SAM 2 Hand-Inpainted Clean Video — Push (`masked_clean_push.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/masked_clean_push.mp4)
  - [Raw Human Recording — Pick & Place (`raw_pick_and_place.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/raw_pick_and_place.mp4)
  - [Raw Human Recording — Push (`raw_push.mp4`)](https://huggingface.co/datasets/omaar97/humanoid_challenge_dataset/resolve/main/media/raw_push.mp4)
- **Fine-Tuned SmolVLA Checkpoint (`omaar97/smolvla_humanoid_challenge`)**: [https://huggingface.co/omaar97/smolvla_humanoid_challenge](https://huggingface.co/omaar97/smolvla_humanoid_challenge)

---

## 1. Pipeline Overview

```
Raw Human Videos (raw_pick_and_place.mp4, raw_push.mp4)
   │
   ├─► [1. src/camera_params.py] ────────────► Multi-frame ArUco PnP calibration (camera_extrinsics.npz)
   │
   ├─► [2. src/process_hand_segmentation.py] ► MediaPipe + SAM 2 hand removal & background inpainting
   │                                           (masked_clean_pick_and_place.mp4, masked_clean_push.mp4)
   │
   ├─► [3. src/overlay_franka_mujoco.py] ────► 3D hand/cup retargeting + MuJoCo Franka FR3 rendering
   │                                           (franka_overlay_pick_and_place.mp4, franka_overlay_push.mp4)
   │
   ├─► [4. src/create_lerobot_dataset.py] ───► Episode segmentation + 640x480 crop + 4D EEF LeRobot v3.0 dataset
   │                                           (lerobot_dataset/: 50 episodes, 4,711 frames @ 30 FPS)
   │
   ├─► [5. SmolVLA Fine-Tuning (Colab)] ─────► Fine-tune lerobot/smolvla_base on 4D EEF state/action + RGB + task prompt
   │
   └─► [6. src/compare_controllers.py] ─────► Multi-controller benchmark in MuJoCo
                                               (controller_comparison_results.md)
```

### Tasks & Language Prompts
1. **Pick and Place** (`25` training episodes, `10` evaluation scenarios):
   - Prompt: `"pick the green cup and place it on top of the pink box"`
2. **Push** (`25` training episodes, `10` evaluation scenarios):
   - Prompt: `"push the green cup next to the pink box"`

---

## 2. Pipeline Details

To make the pipeline choices and implementation transparent, below is a breakdown of what is solved generally, with a focus on the parts that have been simplified or hard-coded.

Hard-Coded parts were mostly generated by the LLM agents according to the given instructions. The overall goal was to ensure the human arm segmentation, robot overlay, and gripper state are suitable for the final dataset. Human verification was made by checking the overall code flow and the output videos, rather than focusing on exact bits of the code and its implementation efficiency. There have been a few iterations before reaching the final, suitable videos.

### A. Camera Parameters Extraction

1. **Camera Intrinsics**: 
   - These were obtained from the Android app camera2test. They were scaled according to the video frame resolution.
2. **ArUco Marker**:
   - The 0 id from the 4*4 ArUco marker family was placed on the table to aid the camera extrinsics calibration.
3. **Multi-Frame Camera Extrinsics Calibration**:
   - Uses `cv2.SOLVEPNP_IPPE_SQUARE` across 60 frames and projects the mean rotation matrix onto $SO(3)$ via SVD, resuling in the .npz file output.

---

### B. Dataset creation with hand segmentation via SAM2 and Franka robot overlay

1. **MediaPipe for Hand Extraction**:
   - The scripts use MediaPipe to initialize the SAM2 segmentation and to also extract rotation vectors and the expected 3D pose of the robot, which is used for both the robot overlay and the dataset creation.
2. **SAM2 for Hand and Arm Segmentation**
   - The choice for SAM2 was to have it more robust than simple image processing segmentation. This was followed by a few image processing techniques (e.g., Gaussian blur, morphological operations, etc) to enhance the output.
3. **Fixed Wrist-Up End-Effector Orientation**:
   - When doing the Franka inverse kinematics, the wrist was always fixed with wrist-up configuration. The human hand did not significantly rotate in the collected dataset, so this choice was made to simplify the kinematics of the robot. Consequently, the robot state and prediction coming from the dataset were 4D vectors (3D position + gripper), as the orientation was fixed.
4. **Custom Gripper**:
   - The gripper used in the task is similar to the Franka Hand, but its parameters are changed to account for simplified grasping of the cup.
5. **Color-based Segmentation**:
   - The cup and the box were segmented using conventional color-based thresholding. These information were used to facilitate dataset creation across several operations (e.g., pink box info to enhance z-plane measurements, green cup info combined with mujoco depth info to prevent occlusion in the overlay).
6. **Binary Gripper Actuation**:
   - The gripper command is set either as open or close in the dataset labels.
7. **Further Minor Processing**:
   - There were other steps before feeding the dataset to the SmolVLA model, which mainly involved things like splitting the videos into episodes, cutting the ArUco marker out of the frame, resizing the images, and other standard data processing practices. One major aspect is that the LLM agent decided to extend the joint limits of the robot instead of changing the base location to improve the overlay accuracy. This was left as it is in the training as the camera only takes the image of the scene with the robot gripper near the object, rather than full third-person image. In execution, this was rectified by changing the robot base and maintaining the original robot parameters. 

---


### C. Training:

1. **Google Colab**:
   - A100 GPU on Google Colab was used to train the model.
2. **Local Inference**:
   - Locally, an RTX 2070 mobile GPU was used for inference during testing.

---


### D. MuJoCo Environment Testing & Implementation Notes:

1. **Scene Approximations & Collision Geometry**:
   - To match the real camera viewpoint with ArUco marker extrinsics, table objects are modeled using primitive collision geometries (the variable cross-section paper cup is represented as a uniform cylinder scaled by ~0.72x). Collision bodies and visual placements were calibrated so visual real-camera rendering aligns with training demonstrations.
2. **Cup Orientation Constraint**:
   - In rigid-body simulation with planar rubber finger pads, grasping a cylinder with two contact points easily produces high contact torque chattering and unphysical pendulum tilting. To stabilize object transport, `step()` fixes the cup quaternion upright (`[1.0, 0.0, 0.0, 0.0]`) and zeroes angular velocities whenever `phys_grip == CLOSED` and finger opening `f_open > 0.010m`. This constraint remains active throughout grasped transport and placement.
3. **Approach Clearance Z-Offset (Pick & Place)**:
   - When the human teleoperation trajectories were retargeted to the Franka FR3 end-effector, the resulting 3D trajectories traversed slightly too low relative to the 5 cm pink box rim. In rigid MuJoCo simulation, this caused the cup base to collide with the front edge of the box. To clear the box rim during transfer, `step()` smoothly blends an exponential moving average offset of up to `+2.2 cm` (`target_z_off = 0.022`) to the IK controller's target height while crossing the box boundary (`x > 0.275, y > -0.035`). To avoid an out-of-distribution observation shift to the policy, `get_state_4d()` subtracts this offset from `observation.state[2]`.
4. **Push vs. Pick Starting Pose & Generalization Scope**:
   - **Pick-and-Place**: The robot arm always starts from the *fixed* rest pose `[0.385, -0.105, 0.048]` regardless of scenario. The policy must visually locate the cup across all 10 table locations and navigate toward it from scratch.
   - **Push**: The arm is always initialized 7.5 cm behind the cup (`[cx, cy - 0.075, 0.048]`). The push policy therefore executes a forward push stroke from a pre-aligned pose rather than acquiring the cup from arbitrary locations.
   - Note that the cup location is always varied across the scenarios.
5. **Pink Box Dynamics**:
   - In the training set, the pink box remained stationary on the table. In the primary test environment (`FrankaCupMuJoCoEnv`), the pink box is static. For users who wish to test disturbance reactions, [`src/mujoco_dynamic_box_test_env.py`](src/mujoco_dynamic_box_test_env.py) provides a fully movable 6-DOF box with relative success metrics.
6. **Physics Simulation & Control Frequency**:
   - The MuJoCo physics engine runs with `timestep = 0.002 s` (500 Hz physics). With 17 substeps per control cycle, the effective policy control rate is $500 / 17 \approx 29.41\text{ Hz}$ (closely approximating 30 Hz). The reported `tau_norm` metric captures the instantaneous motor torque norm at the final substep of each control cycle.


---

## 3. Repository Structure

| File / Directory | Description |
| :--- | :--- |
| [`src/camera_params.py`](src/camera_params.py) | Intrinsics scaling (`4096×3072` $\to$ `1920×1080`) and multi-frame ArUco PnP extrinsics calibration (`camera_extrinsics.npz`). |
| [`src/process_hand_segmentation.py`](src/process_hand_segmentation.py) | MediaPipe + SAM 2 (`sam2.1_hiera_tiny.pt`) streaming hand/arm segmentation and background inpainting (`masked_clean_*.mp4`). |
| [`src/overlay_franka_mujoco.py`](src/overlay_franka_mujoco.py) | Retargets human hand trajectories to the 3D Franka FR3 arm in MuJoCo and renders overlaid robot videos (`franka_overlay_*.mp4`). |
| [`src/create_lerobot_dataset.py`](src/create_lerobot_dataset.py) | Segments the continuous videos into 50 active episodes (25 pick-and-place + 25 push), crops/resizes to `640×480`, and builds `lerobot_dataset/` (LeRobot v3.0). |
| [`src/mujoco_main_test_env.py`](src/mujoco_main_test_env.py) | Pure MuJoCo evaluation environment (`FrankaCupMuJoCoEnv`), 20 standardized test scenarios, and SmolVLA rollout evaluator. |
| [`src/mujoco_trajectory_viewer.py`](src/mujoco_trajectory_viewer.py) | Interactive MuJoCo passive viewer & video visualizer for SmolVLA predicted trajectories with language prompt selection (`--prompt`). |
| [`src/mujoco_velocity_controller_test.py`](src/mujoco_velocity_controller_test.py) | Task-space differential velocity controller test executing predicted SmolVLA trajectories via damped least-squares differential IK. |
| [`src/mujoco_impedance_controller_test.py`](src/mujoco_impedance_controller_test.py) | End-Effector Cartesian impedance controller test with analytical gravity & Coriolis compensation. |
| [`src/validate_pipeline.py`](src/validate_pipeline.py) | Standalone pipeline validation script for scripted trajectory execution and physics checks. |
| [`src/compare_controllers.py`](src/compare_controllers.py) | Comprehensive side-by-side comparison benchmark evaluating Position, Velocity, and Impedance controllers across all 10 scenes per task, outputting comparison matrices (`controller_comparison_results.md`). |
| [`src/mujoco_dynamic_box_test_env.py`](src/mujoco_dynamic_box_test_env.py) | Experimental MuJoCo test environment with a movable 6-DOF pink box that reacts physically to collisions. |
| [`src/download_assets.py`](src/download_assets.py) | Helper script to download the fine-tuned SmolVLA model (`--model`), videos (`--videos`), and LeRobot dataset (`--dataset`) from Hugging Face. |
| [`Dockerfile`](Dockerfile) | Containerized environment with CUDA, headless MuJoCo (`egl`/`osmesa`), dependencies, and optional build-time Hugging Face asset fetching. |
| [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) | VS Code **Reopen in Container** configuration (builds in lightweight mode by default). |
| [`franka_fr3/`](franka_fr3/) | Unmodified Franka Emika FR3 MJCF model (`fr3.xml`) and visual/collision meshes (`assets/`). |
| [`camera_extrinsics.npz`](camera_extrinsics.npz) | Calibrated camera intrinsics `K` and camera-to-table extrinsics (`R_marker_cam`, `t_marker_cam`). |

---

## 4. Step-by-Step Reproduction Guide

A **Docker** container is provided to facilitate reproducibility. Make sure Docker is installed and configured to use the GPU. The next steps have been reproduced after building the container via **VS Code Dev Containers** to use its the integrated terminal directly. They may require minor modifications if used outside this building method.

> [!NOTE]
> **Automatic Raw Video Download**:
> During the Docker build, the raw human demonstration videos (`raw_pick_and_place.mp4` and `raw_push.mp4`) are automatically downloaded from Hugging Face into the `media/` directory (`media/raw_pick_and_place.mp4`, `media/raw_push.mp4`). This is required to start reproducing the results from the first step.

---

### Quickstart Benchmark (Run Inside Container Terminal)

Once inside the container terminal, simply execute:

```bash
# 1. Run the 20-scenario SmolVLA evaluation (auto-downloads checkpoint from Hugging Face if not local):
python3 src/mujoco_main_test_env.py --checkpoint omaar97/smolvla_humanoid_challenge --output-video outputs/smolvla_eval_rollouts.mp4

# 2. Run the automated 3-controller comparison benchmark across all 20 scenarios:
python3 src/compare_controllers.py --checkpoint omaar97/smolvla_humanoid_challenge --output-markdown outputs/comparison_results.md
```

---

### Executing Full Pipeline Steps

Every stage of the data processing, calibration, and evaluation pipeline is executed directly inside the container terminal:

#### Step 1: ArUco Camera Calibration
Computes camera-to-table transformation (`R_marker_cam`, `t_marker_cam`, `K`) across 60 frames from `media/raw_pick_and_place.mp4` and outputs `camera_extrinsics.npz`:
```bash
python3 src/camera_params.py
```

#### Step 2: SAM 2 Hand Inpainting & Background Clean Video Generation
Uses MediaPipe to prompt SAM 2 (`sam2.1_hiera_small.pt`) and replaces the human arm with clean background:
```bash
python3 src/process_hand_segmentation.py --video media/raw_push.mp4 --output media/masked_clean_push.mp4
```
```bash
python3 src/process_hand_segmentation.py --video media/raw_pick_and_place.mp4 --output "media/masked_clean_pick_and_place.mp4"
```
*Outputs*: `media/masked_clean_pick_and_place.mp4`, `media/masked_clean_push.mp4`.

#### Step 3: Trajectory Retargeting & Franka FR3 Overlay
Extracts 3D end-effector trajectories, solves inverse kinematics, and composites rendered Franka FR3 arm onto clean video:
```bash
python3 src/overlay_franka_mujoco.py --task all
```
*Outputs*: `media/franka_overlay_pick_and_place.mp4`, `media/franka_overlay_push.mp4`.

#### Step 4: Build LeRobot v3.0 Dataset
Splits recordings in `media/` into 50 episodes (`25` Pick-and-Place + `25` Push), crops out `x < 580`, resizes to `640×480`, and writes the lerobot dataset format:
```bash
python3 src/create_lerobot_dataset.py
```
*Outputs*: `lerobot_dataset/`.

#### Step 5: Fine-Tune SmolVLA (`lerobot/smolvla_base`)
Training was conducted on an A100 GPU on Google Colab directly against the public Hugging Face dataset (`omaar97/humanoid_challenge_dataset`) following roughly these steps (saved and mounted on a Google Drive folder, arg omitted here):
```bash
!pip install "lerobot[smolvla]==0.4.4" "transformers==4.57.6" num2words
!lerobot-train \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=omaar97/humanoid_challenge_dataset \
  --steps=100000 \
  --save_freq=10000 \
  --wandb.enable=false
```
Fine-tuned model weights are hosted on Hugging Face: [`omaar97/smolvla_humanoid_challenge`](https://huggingface.co/omaar97/smolvla_humanoid_challenge).

#### Step 6: Closed-Loop Multi-Controller Benchmark
Benchmarks Position, Velocity, and Impedance controllers across all 20 standardized test scenes:
```bash
python3 src/compare_controllers.py --checkpoint omaar97/smolvla_humanoid_challenge
```

---

## 5. Benchmark Results: Multi-Controller Evaluation Across 20 MuJoCo Scenarios

> [!NOTE]
> **Controller Tuning & Proof-of-Concept Disclaimer**:
> The differential velocity and Cartesian impedance controllers have **not been tuned** and their control parameters were evaluated zero-shot on the first attempt without gain optimization ($K_v$, $K_p$, $D$). Consequently, this benchmark is not a fair head-to-head comparison; it simply serves as a showcase for using SmolVLA to drive low-level torque-based velocity and impedance control laws out of the box.
> The degrees in the results show the tilt of the cup after the task is finished.

### Controller Comparison Matrix

```text
===================================================================================================
                    SMOLVLA CONTROLLER COMPARISON MATRIX
===================================================================================================
Scenario                   | Task  |  Position Control  |  Velocity Control  | Impedance Control  
---------------------------------------------------------------------------------------------------
pick_01_nominal_center     | Pick  |    PASS (0.1°)     |    PASS (0.7°)     |    PASS (0.1°)     
pick_02_front_closer       | Pick  |    PASS (0.2°)     |    PASS (1.1°)     |    FAIL (0.0°)     
pick_03_upper_near_wall    | Pick  |    PASS (0.3°)     |    PASS (0.2°)     |    FAIL (0.0°)     
pick_04_far_right          | Pick  |    PASS (0.2°)     |    FAIL (13.6°)    |    FAIL (89.9°)    
pick_05_near_box_right     | Pick  |    PASS (0.1°)     |    PASS (0.4°)     |    PASS (0.1°)     
pick_06_mid_right          | Pick  |    PASS (0.4°)     |    PASS (1.0°)     |    FAIL (0.0°)     
pick_07_upper_right        | Pick  |    PASS (0.2°)     |    PASS (0.1°)     |    PASS (0.4°)     
pick_08_deep_wall          | Pick  |    FAIL (0.0°)     |    FAIL (0.0°)     |    FAIL (0.1°)     
pick_09_left_mid           | Pick  |    PASS (0.2°)     |    PASS (0.9°)     |    PASS (0.0°)     
pick_10_center_back        | Pick  |    PASS (0.3°)     |    PASS (0.2°)     |    PASS (0.3°)     
push_01_nominal_front      | Push  |    PASS (0.0°)     |    PASS (0.0°)     |    PASS (0.2°)     
push_02_left_front         | Push  |    PASS (0.1°)     |    PASS (0.0°)     |    PASS (0.0°)     
push_03_far_right_front    | Push  |    PASS (0.0°)     |    PASS (0.0°)     |    PASS (2.3°)     
push_04_deep_front         | Push  |    PASS (0.0°)     |    PASS (0.1°)     |    PASS (1.6°)     
push_05_mid_table          | Push  |    PASS (0.0°)     |    PASS (0.1°)     |    PASS (0.0°)     
push_06_mid_right_front    | Push  |    PASS (0.0°)     |    PASS (0.1°)     |    PASS (3.0°)     
push_07_inner_left         | Push  |    PASS (0.0°)     |    PASS (0.0°)     |    PASS (0.0°)     
push_08_shallow_stroke     | Push  |    PASS (0.0°)     |    PASS (0.1°)     |    PASS (0.2°)     
push_09_outer_right_deep   | Push  |    PASS (0.0°)     |    PASS (0.0°)     |    PASS (3.5°)     
push_10_center_front       | Push  |    PASS (0.0°)     |    PASS (0.0°)     |    PASS (0.0°)     
---------------------------------------------------------------------------------------------------
SUMMARY PERFORMANCE:
---------------------------------------------------------------------------------------------------
Pick & Place Success Rate          |    9/10 (90.0%)    |    8/10 (80.0%)    |    5/10 (50.0%)    
Push Success Rate                  |   10/10 (100.0%)   |   10/10 (100.0%)   |   10/10 (100.0%)   
Overall Success Rate               |   19/20 (95.0%)    |   18/20 (90.0%)    |   15/20 (75.0%)    
Mean Tracking Error                |  N/A (Pos Servo)   |       0.2 cm       |       1.5 cm       
Mean Torque Effort ||tau||         |  N/A (Kinematic)   |      33.1 Nm       |      32.1 Nm       
===================================================================================================
```

---

## 6. MuJoCo Simulation Environments & Controllers

### 1. Main MuJoCo Test Environment — Position Controller (`mujoco_main_test_env.py`)
Primary evaluation environment implementing the Franka FR3 arm with numerical IK control. Evaluates the SmolVLA policy across the 20 standardized test scenes. It automatically resolves local `last` or `pretrained_model` checkpoint, or streams from Hugging Face Hub (`omaar97/smolvla_humanoid_challenge`).

```bash
# Evaluate policy across all 20 scenarios (headless, writes smolvla_eval_rollouts.mp4)
python3 src/mujoco_main_test_env.py

# Launch in interactive MuJoCo 3D passive viewer GUI
python3 src/mujoco_main_test_env.py --viewer

# Evaluate a specific task or instance
python3 src/mujoco_main_test_env.py --task pick_and_place
python3 src/mujoco_main_test_env.py --task push
python3 src/mujoco_main_test_env.py --instance pick_01_nominal_center --viewer
```

### 2. Interactive 3D Trajectory Viewer (`mujoco_trajectory_viewer.py`)
Visualizes full multi-step SmolVLA predicted trajectories in 3D using `mujoco.viewer.launch_passive` (or simply save them as videos if the viewer is not loading):

```bash
# Launch interactive 3D viewer (default: loads 'last' checkpoint, Pick-and-Place prompt)
python3 src/mujoco_trajectory_viewer.py

# Launch with Push task prompt
python3 src/mujoco_trajectory_viewer.py --prompt "push the green cup next to the pink box"
```

### 3. Task-Space Differential Velocity Controller (`mujoco_velocity_controller_test.py`)
Tracks predicted SmolVLA trajectories using DLS differential kinematics executed at the low level via direct motor torque commands:
- **Control Law**: $`\tau_{\text{cmd}} = K_{v,\text{joint}} (\dot{q}_{\text{des}} - \dot{q}) + \tau_{\text{bias}}`$ where $\tau_{\text{bias}}$ provides analytical gravity and Coriolis compensation (`data.qfrc_bias`).

```bash
# Evaluate SmolVLA policy (Pick & Place)
python3 src/mujoco_velocity_controller_test.py --task pick_and_place

# Evaluate SmolVLA policy (Push)
python3 src/mujoco_velocity_controller_test.py --task push

# Launch in interactive MuJoCo 3D viewer GUI
python3 src/mujoco_velocity_controller_test.py --viewer
```

### 4. End-Effector Cartesian Impedance Controller (`mujoco_impedance_controller_test.py`)
Executes task-space Cartesian impedance control at the End-Effector without nullspace control:
- **Control Law**: $\tau_{\text{cmd}} = J^T W_{\text{task}} + \tau_{\text{bias}}$.

```bash
# Evaluate SmolVLA policy (Pick & Place)
python3 src/mujoco_impedance_controller_test.py --task pick_and_place

# Evaluate SmolVLA policy (Push)
python3 src/mujoco_impedance_controller_test.py --task push

# Launch in interactive MuJoCo 3D viewer GUI
python3 src/mujoco_impedance_controller_test.py --viewer
```

### 5. Multi-Controller Comparison Benchmark (`compare_controllers.py`)
Executes an automated side-by-side benchmark comparing **Position Control**, **Velocity Control**, and **Cartesian Impedance Control** across all 10 testing scenes for each task/prompt:
- Loads the SmolVLA policy once and runs back-to-back evaluations across controllers.
- Computes success rates, average tracking error, and joint torque effort ($||\tau||$).
- Prints a side-by-side comparison matrix directly to the console.
- Generates a markdown report (`controller_comparison_results.md`) and JSON dataset (`controller_comparison_results.json`).

```bash
# Run fast comparison across all 3 controllers on all 20 scenes (No video recording by default)
python3 src/compare_controllers.py

# Compare controllers strictly on Pick-and-Place (all 10 scenes, no video)
python3 src/compare_controllers.py --task pick_and_place

# Compare controllers strictly on Push (all 10 scenes, no video)
python3 src/compare_controllers.py --task push

# Compare on a single scenario (fast headless run)
python3 src/compare_controllers.py --instance pick_01_nominal_center

# (Optional) Synchronized 3-way split-screen video on a specific scenario
python3 src/compare_controllers.py --split-screen --output-video split_comparison.mp4
```

> [!TIP]
> **Optional: Scripted Trajectory Validation (`validate_pipeline.py`)**:
> A file for checking the MuJoCo physics simulation, kinematics, collision geometry, and torque limits independently without requiring SmolVLA:
> ```bash
> # Validate position controller via scripted trajectory
> python3 src/validate_pipeline.py --mode position
> 
> # Validate torque-based velocity and impedance controllers with 3D GUI
> python3 src/validate_pipeline.py --mode velocity --viewer
> python3 src/validate_pipeline.py --mode impedance --viewer
> ```

---

## 7. Local Virtual Environment (`.venv`) (Optional Alternative)

If you prefer to run directly on your host machine without Docker, you can set up a local Python 3.10 virtual environment:

```bash
# 1. Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies:
pip install "numpy==1.26.4" torch torchvision "transformers==4.57.6" \
    num2words "mujoco==3.15.0" "mediapipe==0.10.21" \
    "opencv-contrib-python==4.11.0.86" "opencv-python-headless==4.11.0.86" \
    scipy pandas pyarrow huggingface_hub datasets diffusers accelerate pyserial \
    einops av jsonlines draccus gymnasium torchcodec deepdiff imageio termcolor
pip install --no-deps "rerun-sdk==0.26.2" "lerobot==0.4.4"

# 3. (Optional) Download hosted model weights and assets:
python3 src/download_assets.py --all
```

When using the virtual environment approach, you may need to install other versions or face compatibility issues. **Docker** is recommended and has been verified.
