import cv2
import numpy as np

def build_camera_matrix(video_width: int, video_height: int):
    # Base intrinsics calibrated at 4096 x 3072 (4:3 full sensor)
    base_w, base_h = 4096.0, 3072.0
    fx_base, fy_base = 2770.6487, 2744.9617
    cx_base, cy_base = 2048.0, 1536.0

    # When recording 16:9 video (e.g. 1920x1080) from the 4:3 sensor, the sensor is
    # vertically cropped around the center and uniformly scaled by video_width / base_w.
    scale = video_width / base_w
    crop_offset_y = (base_h * scale - video_height) / 2.0

    camera_matrix = np.array([
        [fx_base * scale, 0.0,             cx_base * scale],
        [0.0,             fy_base * scale, cy_base * scale - crop_offset_y],
        [0.0,             0.0,             1.0]
    ], dtype=np.float64)

    # Skew is 0, assuming minimal distortion
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def _solve_marker_pnp(obj_points, marker_corners, camera_matrix, dist_coeffs):
    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_points,
        marker_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or len(rvecs) == 0:
        return False, None, None

    # Pick the physically valid IPPE branch where marker +Z points toward the camera (Z_cam < 0)
    for rvec, tvec in zip(rvecs, tvecs):
        R_mat, _ = cv2.Rodrigues(rvec)
        if R_mat[2, 2] < 0 and R_mat[1, 2] < 0:
            return True, R_mat, tvec

    R_mat, _ = cv2.Rodrigues(rvecs[0])
    return True, R_mat, tvecs[0]


def estimate_camera_extrinsics(image, marker_length_m=0.08):
    h, w = image.shape[:2]
    camera_matrix, dist_coeffs = build_camera_matrix(w, h)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or 0 not in ids:
        raise ValueError("ArUco Marker ID 0 not found.")

    idx = np.where(ids.flatten() == 0)[0][0]
    marker_corners = corners[idx]

    half = marker_length_m / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float64)

    success, R_cam_marker, tvec = _solve_marker_pnp(
        obj_points, marker_corners, camera_matrix, dist_coeffs
    )
    if not success:
        raise RuntimeError("Failed to solve PnP for ArUco marker.")

    # Camera position in the table/marker coordinate frame
    R_marker_cam = R_cam_marker.T
    t_marker_cam = -R_marker_cam @ tvec

    return camera_matrix, R_cam_marker, tvec, R_marker_cam, t_marker_cam


def calibrate_from_video_multiframe(
    video_path,
    num_frames=60,
    marker_length_m=0.08,
    save_path="camera_extrinsics.npz"
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_matrix, dist_coeffs = build_camera_matrix(w, h)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    half = marker_length_m / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float64)

    tvecs = []
    rot_matrices = []

    frame_count = 0
    while frame_count < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None or 0 not in ids:
            continue

        idx = np.where(ids.flatten() == 0)[0][0]
        marker_corners = corners[idx]

        success, R_mat, tvec = _solve_marker_pnp(
            obj_points, marker_corners, camera_matrix, dist_coeffs
        )
        if success:
            rot_matrices.append(R_mat)
            tvecs.append(tvec.flatten())

    cap.release()

    if len(tvecs) < 10:
        raise ValueError(f"Too few valid detections ({len(tvecs)}/{num_frames}) to aggregate extrinsics.")

    tvecs = np.array(tvecs)

    # Sanity check: std deviation in translation (should be under 2-3 mm on a stationary tripod)
    std_tvec_mm = np.std(tvecs, axis=0) * 1000.0
    print(f"Translation jitter std dev (mm): X={std_tvec_mm[0]:.2f}, Y={std_tvec_mm[1]:.2f}, Z={std_tvec_mm[2]:.2f}")

    # 1. Translation: robust median across frames
    median_tvec = np.median(tvecs, axis=0).reshape(3, 1)

    # 2. Rotation: average rotation matrix projected back to valid SO(3) via SVD
    mean_R = np.mean(rot_matrices, axis=0)
    U, _, Vt = np.linalg.svd(mean_R)
    R_cam_marker = U @ Vt
    if np.linalg.det(R_cam_marker) < 0:
        R_cam_marker[:, -1] *= -1  # Ensure valid right-handed coordinate system

    # Invert to express the camera position and orientation in marker/table space
    R_marker_cam = R_cam_marker.T
    t_marker_cam = -R_marker_cam @ median_tvec

    np.savez(
        save_path,
        K=camera_matrix,
        R_cam_marker=R_cam_marker,
        t_cam_marker=median_tvec,
        R_marker_cam=R_marker_cam,
        t_marker_cam=t_marker_cam
    )
    print(f"Aggregated extrinsics computed over {len(tvecs)} frames and saved to {save_path}")
    return camera_matrix, R_cam_marker, median_tvec, R_marker_cam, t_marker_cam

if __name__ == "__main__":
    # Run once per camera setup session
    calibrate_from_video_multiframe("raw_pick_and_place.mp4")