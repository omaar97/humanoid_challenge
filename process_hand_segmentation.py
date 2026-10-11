import cv2
import mediapipe as mp
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def process_video_with_delayed_hand(
    video_path,
    output_video_path="masked_clean.mp4",
    checkpoint="sam2_repo/checkpoints/sam2.1_hiera_small.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )

    entry_frame_idx = None
    prompt_points = None
    frame_idx = 0

    # Detect the first hand frame before creating SAM 2's video state.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            wrist = [landmarks[0].x * width, landmarks[0].y * height]
            mcp_mid = [landmarks[9].x * width, landmarks[9].y * height]
            mcp_index = [landmarks[5].x * width, landmarks[5].y * height]
            arm_vec = np.array(wrist) - np.array(mcp_mid)
            arm_vec /= np.linalg.norm(arm_vec) + 1e-6
            forearm_point = (np.array(wrist) + arm_vec * 80).tolist()
            entry_frame_idx = frame_idx
            prompt_points = np.array(
                [wrist, mcp_mid, mcp_index, forearm_point], dtype=np.float32
            )
            break
        frame_idx += 1

    cap.release()
    mp_hands.close()

    if entry_frame_idx is None:
        raise RuntimeError("No hand was detected in the video")

    print(
        f"Hand detected at frame {entry_frame_idx} "
        f"(~{entry_frame_idx / fps:.2f}s). Initializing SAM 2."
    )

    inference_state = predictor.init_state(
        video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    labels = np.ones(len(prompt_points), dtype=np.int32)
    _, object_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=entry_frame_idx,
        obj_id=1,
        points=prompt_points,
        labels=labels,
        normalize_coords=False,
    )

    propagation = predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=entry_frame_idx,
        reverse=False,
    )

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap = cv2.VideoCapture(video_path)
    current_output = (entry_frame_idx, object_ids, mask_logits)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < entry_frame_idx:
            out.write(frame)
        elif current_output is not None:
            propagated_frame_idx, _, propagated_masks = current_output
            if propagated_frame_idx != frame_idx:
                raise RuntimeError(
                    f"SAM 2 returned frame {propagated_frame_idx}, "
                    f"expected {frame_idx}"
                )

            mask = (
                (propagated_masks[0] > 0.0)
                .cpu()
                .numpy()
                .squeeze()
                .astype(np.uint8)
                * 255
            )
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            clean_frame = cv2.inpaint(
                frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
            )
            out.write(clean_frame)

            try:
                current_output = next(propagation)
            except StopIteration:
                current_output = None
        else:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    process_video_with_delayed_hand(
        video_path="raw_pick_and_place.mp4",
        output_video_path="masked_clean.mp4",
    )

