# SmolVLA Controller Comparison Report

Evaluated on fine-tuned checkpoint: `/home/omaar97/humanoid_challenge/last-20261012T095114Z-1/last/pretrained_model`

## 1. Summary Performance Metrics

| Metric | Position Control | Velocity Control | Impedance Control |
| :--- | :---: | :---: | :---: |
| **Pick & Place Success** | **9/10 (90.0%)** | **8/10 (80.0%)** | **5/10 (50.0%)** |
| **Push Success** | **10/10 (100.0%)** | **10/10 (100.0%)** | **10/10 (100.0%)** |
| **Overall Success** | **19/20 (95.0%)** | **18/20 (90.0%)** | **15/20 (75.0%)** |
| **Mean Tracking Error** | N/A (Kinematic) | 0.2 cm | 1.5 cm |
| **Mean Torque Effort ||tau||** | N/A | 33.1 Nm | 32.1 Nm |

## 2. Per-Scenario Comparison Table

| Scenario | Task | Position Control | Velocity Control | Impedance Control |
| :--- | :--- | :---: | :---: | :---: |
| `pick_01_nominal_center` | Pick & Place | **PASS** (tilt=0.1°) | **PASS** (tilt=0.7°) | **PASS** (tilt=0.1°) |
| `pick_02_front_closer` | Pick & Place | **PASS** (tilt=0.2°) | **PASS** (tilt=1.1°) | FAIL (tilt=0.0°) |
| `pick_03_upper_near_wall` | Pick & Place | **PASS** (tilt=0.3°) | **PASS** (tilt=0.2°) | FAIL (tilt=0.0°) |
| `pick_04_far_right` | Pick & Place | **PASS** (tilt=0.2°) | FAIL (tilt=13.6°) | FAIL (tilt=89.9°) |
| `pick_05_near_box_right` | Pick & Place | **PASS** (tilt=0.1°) | **PASS** (tilt=0.4°) | **PASS** (tilt=0.1°) |
| `pick_06_mid_right` | Pick & Place | **PASS** (tilt=0.4°) | **PASS** (tilt=1.0°) | FAIL (tilt=0.0°) |
| `pick_07_upper_right` | Pick & Place | **PASS** (tilt=0.2°) | **PASS** (tilt=0.1°) | **PASS** (tilt=0.4°) |
| `pick_08_deep_wall` | Pick & Place | FAIL (tilt=0.0°) | FAIL (tilt=0.0°) | FAIL (tilt=0.1°) |
| `pick_09_left_mid` | Pick & Place | **PASS** (tilt=0.2°) | **PASS** (tilt=0.9°) | **PASS** (tilt=0.0°) |
| `pick_10_center_back` | Pick & Place | **PASS** (tilt=0.3°) | **PASS** (tilt=0.2°) | **PASS** (tilt=0.3°) |
| `push_01_nominal_front` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) | **PASS** (tilt=0.2°) |
| `push_02_left_front` | Push | **PASS** (tilt=0.1°) | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) |
| `push_03_far_right_front` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) | **PASS** (tilt=2.3°) |
| `push_04_deep_front` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.1°) | **PASS** (tilt=1.6°) |
| `push_05_mid_table` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.1°) | **PASS** (tilt=0.0°) |
| `push_06_mid_right_front` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.1°) | **PASS** (tilt=3.0°) |
| `push_07_inner_left` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) |
| `push_08_shallow_stroke` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.1°) | **PASS** (tilt=0.2°) |
| `push_09_outer_right_deep` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) | **PASS** (tilt=3.5°) |
| `push_10_center_front` | Push | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) | **PASS** (tilt=0.0°) |
