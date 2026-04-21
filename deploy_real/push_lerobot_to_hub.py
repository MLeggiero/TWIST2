"""Re-push an existing on-disk LeRobot dataset to HuggingFace Hub.

Useful when conversion succeeded but ``--push_to_hub`` failed during a prior run.
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. user/dataset_name)")
    parser.add_argument("--root", type=str, default=None,
                        help="Local dataset directory "
                             "(default: ~/.cache/huggingface/lerobot/<repo_id>)")
    parser.add_argument("--public", action="store_true", default=False,
                        help="Push as a public repo (default is private)")
    parser.add_argument("--no_videos", action="store_true", default=False,
                        help="Skip uploading video files")
    parser.add_argument("--upload_large_folder", action="store_true", default=False,
                        help="Use HfApi.upload_large_folder() for very large datasets")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root or str(Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
    print(f"Loading dataset from {root}")
    ds = LeRobotDataset(repo_id=args.repo_id, root=root)
    print(f"Pushing to hub: {args.repo_id}  (private={not args.public})")
    ds.push_to_hub(
        private=not args.public,
        push_videos=not args.no_videos,
        upload_large_folder=args.upload_large_folder,
    )
    print("Done.")


if __name__ == "__main__":
    main()
