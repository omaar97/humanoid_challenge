from collections import OrderedDict
import contextlib
import os
from queue import Queue
import sys
from threading import Thread
from unittest.mock import patch
import cv2
import mediapipe as mp
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import load_video_frames as _orig_load_video_frames


class LazyVideoFrameLoader:
    """
    Lazily loads and preprocesses video frames on demand from an MP4 file using OpenCV
    sequential streaming with a bounded LRU cache, avoiding loading the full video into memory.
    """

    def __init__(
        self,
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
        cache_size=32,
    ):
        self.video_path = video_path
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.compute_device = compute_device
        self.cache_size = max(1, cache_size)
        self.scale = (1.0 / (255.0 * img_std)).to(dtype=torch.float32)
        self.bias = (-img_mean / img_std).to(dtype=torch.float32)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError(f"Failed to read first frame from: {video_path}")

        self.video_height, self.video_width = first_frame.shape[:2]
        self.pos = 0
        self.cache = OrderedDict()
        self.last_valid_tensor = self._frame_to_tensor(first_frame)
        self.cache[0] = self.last_valid_tensor

    def _frame_to_tensor(self, frame_bgr):
        resized = cv2.resize(
            frame_bgr,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32)
        img.mul_(self.scale).add_(self.bias)
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        return img

    def __getitem__(self, index):
        if index < 0:
            index += self.num_frames
        if index < 0 or index >= self.num_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self.num_frames})"
            )

        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]

        if index < self.pos:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.pos = -1

        decode_start = max(self.pos + 1, index - self.cache_size + 1)
        for i in range(self.pos + 1, index + 1):
            if i >= decode_start:
                ret, frame = self.cap.read()
                if not ret:
                    return self.last_valid_tensor
                self.pos = i
                tensor = self._frame_to_tensor(frame)
                self.last_valid_tensor = tensor
                self.cache[i] = tensor
                self.cache.move_to_end(i)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            else:
                ret = self.cap.grab()
                if not ret:
                    return self.last_valid_tensor
                self.pos = i

        return self.cache[index]

    def __len__(self):
        return self.num_frames

    def __del__(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()


def _lazy_load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
):
    if isinstance(video_path, str) and not os.path.isdir(video_path):
        img_mean_t = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std_t = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        lazy_images = LazyVideoFrameLoader(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean_t,
            img_std=img_std_t,
            compute_device=compute_device,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    return _orig_load_video_frames(
        video_path=video_path,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=async_loading_frames,
        compute_device=compute_device,
    )


def inpaint_hand_region(frame, mask_bool, dilate_kernel):
    """Dilate the binary hand/arm mask and inpaint the region using fast multi-scale Telea."""
    if mask_bool is None or not np.any(mask_bool):
        return frame

    mask_u8 = cv2.dilate(
        mask_bool.astype(np.uint8) * 255, dilate_kernel, iterations=1
    )
    ys, xs = np.nonzero(mask_u8)
    if len(ys) == 0:
        return frame

    h, w = frame.shape[:2]
    pad = 20
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + 1 + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + 1 + pad)

    roi_frame = frame[y1:y2, x1:x2]
    roi_mask = mask_u8[y1:y2, x1:x2]
    roi_h, roi_w = roi_frame.shape[:2]

    small_w = max(1, int(round(roi_w * 0.25)))
    small_h = max(1, int(round(roi_h * 0.25)))
    small_frame = cv2.resize(
        roi_frame, (small_w, small_h), interpolation=cv2.INTER_AREA
    )
    small_mask = cv2.resize(
        roi_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST
    )
    inpainted_small = cv2.inpaint(
        small_frame, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
    )
    inpainted_roi = cv2.resize(
        inpainted_small, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC
    )

    clean_frame = frame.copy()
    roi_out = clean_frame[y1:y2, x1:x2]
    mask_cond = roi_mask > 0
    roi_out[mask_cond] = inpainted_roi[mask_cond]
    return clean_frame


def resolve_sam2_checkpoint(checkpoint_path: str = "sam2_repo/checkpoints/sam2.1_hiera_small.pt") -> str:
    """Resolve SAM 2 checkpoint from common local paths or auto-download from Meta CDN."""
    candidates = [
        checkpoint_path,
        "checkpoints/sam2.1_hiera_small.pt",
        "sam2_repo/checkpoints/sam2.1_hiera_small.pt",
        "/opt/sam2/checkpoints/sam2.1_hiera_small.pt",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c

    target_dir = os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else "checkpoints"
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, os.path.basename(checkpoint_path) or "sam2.1_hiera_small.pt")
    if not os.path.isfile(target_file):
        print(f"[SAM 2] Checkpoint not found locally. Downloading sam2.1_hiera_small.pt from Meta CDN ...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        urllib.request.urlretrieve(url, target_file)
        print(f"[SAM 2] Checkpoint saved to {target_file}")
    return target_file


def resolve_video_path(filename: str) -> str:
    """Resolve video path preferring media/ directory, then local root and /opt/media."""
    base = os.path.basename(filename)
    candidates = [
        os.path.join("media", base),
        filename,
        base,
        os.path.join("/opt/media", base),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return os.path.join("media", base)


def process_video_with_hand(
    video_path="media/raw_push.mp4",
    output_video_path="media/masked_clean_push.mp4",
    checkpoint="sam2_repo/checkpoints/sam2.1_hiera_small.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
):
    video_path = resolve_video_path(video_path)
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    checkpoint = resolve_sam2_checkpoint(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[process_hand_segmentation] Input video: {video_path}")
    print(f"[process_hand_segmentation] Output video: {output_video_path}")
    print(f"Using device: {device}")
    predictor = build_sam2_video_predictor(
        model_cfg,
        checkpoint,
        device=device,
        hydra_overrides_extra=["++model.fill_hole_area=0"],
    )

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
            forearm_point = np.clip(
                np.array(wrist) + arm_vec * 80, [0, 0], [width - 1, height - 1]
            ).tolist()
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

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast_ctx:
        with patch(
            "sam2.sam2_video_predictor.load_video_frames",
            _lazy_load_video_frames,
        ):
            inference_state = predictor.init_state(
                video_path,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )
        labels = np.ones(len(prompt_points), dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=entry_frame_idx,
            obj_id=1,
            points=prompt_points,
            labels=labels,
            normalize_coords=True,
        )

        # Propagate slightly backward from entry_frame_idx to segment the fingertips
        # entering the frame before MediaPipe reaches its 0.7 detection confidence.
        pre_entry_frames = min(15, entry_frame_idx)
        start_propagate_idx = entry_frame_idx - pre_entry_frames
        early_masks = {}
        if pre_entry_frames > 0:
            for f_idx, _, masks in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=entry_frame_idx,
                max_frame_num_to_track=pre_entry_frames,
                reverse=True,
            ):
                if f_idx < entry_frame_idx:
                    early_masks[f_idx] = (masks[0, 0] > 0.0).cpu().numpy()
            inference_state["output_dict_per_obj"][0][
                "non_cond_frame_outputs"
            ].clear()
            inference_state["frames_tracked_per_obj"][0].clear()

        mask_queue = Queue(maxsize=16)
        writer_error = []

        def writer_worker():
            try:
                dilate_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (21, 21)
                )
                reader = cv2.VideoCapture(video_path)
                writer = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                f_idx = 0
                queue_active = True
                while reader.isOpened():
                    ret, raw_frame = reader.read()
                    if not ret:
                        break

                    if f_idx < start_propagate_idx or not queue_active:
                        writer.write(raw_frame)
                    else:
                        item = mask_queue.get()
                        if item is None:
                            queue_active = False
                            writer.write(raw_frame)
                        else:
                            propagated_frame_idx, mask_bool = item
                            if propagated_frame_idx != f_idx:
                                raise RuntimeError(
                                    f"SAM 2 returned frame {propagated_frame_idx}, "
                                    f"expected {f_idx}"
                                )
                            clean_frame = inpaint_hand_region(
                                raw_frame, mask_bool, dilate_kernel
                            )
                            writer.write(clean_frame)
                    f_idx += 1

                # Drain any remaining items if video ended slightly before queue
                while queue_active:
                    if mask_queue.get() is None:
                        break

                reader.release()
                writer.release()
            except Exception as e:
                writer_error.append(e)

        worker_thread = Thread(target=writer_worker, daemon=True)
        worker_thread.start()

        try:
            for f_idx in range(start_propagate_idx, entry_frame_idx):
                mask_queue.put((f_idx, early_masks.pop(f_idx, None)))

            non_cond_outputs = inference_state["output_dict_per_obj"][0][
                "non_cond_frame_outputs"
            ]
            tracked_frames = inference_state["frames_tracked_per_obj"][0]

            for propagated_frame_idx, _, propagated_masks in (
                predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=entry_frame_idx,
                    reverse=False,
                )
            ):
                if writer_error:
                    raise writer_error[0]

                mask_bool = (propagated_masks[0, 0] > 0.0).cpu().numpy()
                mask_queue.put((propagated_frame_idx, mask_bool))

                # Prune old non-conditioning memory frames beyond SAM 2's temporal window
                # so GPU/CPU memory stays constant across arbitrarily long videos.
                cutoff = propagated_frame_idx - 25
                for old_k in [k for k in non_cond_outputs if k < cutoff]:
                    del non_cond_outputs[old_k]
                for old_k in [k for k in tracked_frames if k < cutoff]:
                    del tracked_frames[old_k]
        finally:
            mask_queue.put(None)
            worker_thread.join()

        if writer_error:
            raise writer_error[0]

    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MediaPipe + SAM 2 hand segmentation & inpainting.")
    parser.add_argument("--video", type=str, default="media/raw_push.mp4", help="Input raw video path.")
    parser.add_argument("--output", type=str, default="media/masked_clean_push.mp4", help="Output inpainted video path.")
    args = parser.parse_args()
    process_video_with_hand(
        video_path=args.video,
        output_video_path=args.output,
    )


