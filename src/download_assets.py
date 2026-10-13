#!/usr/bin/env python3
"""Download optional Hugging Face assets (SmolVLA checkpoint, videos, LeRobot dataset).

Usage examples:
  # Download only the fine-tuned SmolVLA model checkpoint:
  python3 src/download_assets.py --model

  # Download the raw/processed/evaluation videos into the workspace root:
  python3 src/download_assets.py --videos

  # Download the LeRobot v3.0 training dataset into ./lerobot_dataset:
  python3 src/download_assets.py --dataset

  # Download everything (model + videos + dataset):
  python3 src/download_assets.py --all
"""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_DATASET_REPO = "omaar97/humanoid_challenge_dataset"
DEFAULT_MODEL_REPO = "omaar97/smolvla_humanoid_challenge"

RAW_VIDEOS = [
    "raw_pick_and_place.mp4",
    "raw_push.mp4",
]

MEDIA_VIDEOS = [
    "dataset_pick_and_place_25eps.mp4",
    "dataset_push_25eps.mp4",
    "franka_overlay_pick_and_place.mp4",
    "franka_overlay_push.mp4",
    "masked_clean_pick_and_place.mp4",
    "masked_clean_push.mp4",
    "raw_pick_and_place.mp4",
    "raw_push.mp4",
]


def download_model(model_repo: str, output_dir: str = "pretrained_model") -> None:
    print(f"[download_assets] Downloading model from '{model_repo}' -> '{output_dir}/' ...")
    snapshot_download(
        repo_id=model_repo,
        repo_type="model",
        local_dir=output_dir,
    )
    print(f"[download_assets] Model ready at '{output_dir}/'")


def download_videos(dataset_repo: str, output_dir: str = "media", video_list: list = None) -> None:
    target_videos = video_list if video_list is not None else MEDIA_VIDEOS
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for vid in target_videos:
        remote_path = f"media/{vid}"
        target_path = out_root / vid
        try:
            print(f"[download_assets] Fetching '{remote_path}' from '{dataset_repo}' ...")
            cached_file = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=remote_path,
            )
            shutil.copy2(cached_file, target_path)
            print(f"  -> Saved {target_path}")
        except Exception as exc:
            print(f"  [warning] Skipping '{remote_path}' ({exc})")


def download_dataset(dataset_repo: str, output_dir: str = "lerobot_dataset") -> None:
    print(f"[download_assets] Downloading LeRobot dataset from '{dataset_repo}' -> '{output_dir}/' ...")
    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=output_dir,
        ignore_patterns=["media/*"],
    )
    print(f"[download_assets] Dataset ready at '{output_dir}/'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download optional SmolVLA model, videos, and LeRobot dataset from Hugging Face."
    )
    parser.add_argument("--model", action="store_true", help="Download fine-tuned SmolVLA checkpoint.")
    parser.add_argument("--videos", action="store_true", help="Download all pipeline & evaluation MP4 videos.")
    parser.add_argument("--raw-videos", action="store_true", help="Download only raw human videos for reproduction.")
    parser.add_argument("--dataset", action="store_true", help="Download LeRobot v3.0 dataset.")
    parser.add_argument("--all", action="store_true", help="Download model, all videos, and dataset.")
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO)
    parser.add_argument("--dataset-repo", type=str, default=DEFAULT_DATASET_REPO)
    parser.add_argument("--model-dir", type=str, default="pretrained_model")
    parser.add_argument("--media-dir", type=str, default="media")
    args = parser.parse_args()

    if not (args.model or args.videos or args.raw_videos or args.dataset or args.all):
        parser.print_help()
        return

    if args.all or args.model:
        download_model(args.model_repo, args.model_dir)
    if args.all or args.videos:
        download_videos(args.dataset_repo, args.media_dir, MEDIA_VIDEOS)
    elif args.raw_videos:
        download_videos(args.dataset_repo, args.media_dir, RAW_VIDEOS)
    if args.all or args.dataset:
        download_dataset(args.dataset_repo, "lerobot_dataset")


if __name__ == "__main__":
    main()
