"""
PatchedLeRobotDataset: LeRobotDataset subclass that skips video decoding
and returns fake (zero) frames matching the expected video shape.

Usage:
    from patched_lerobot_dataset import PatchedLeRobotDataset

    dataset = PatchedLeRobotDataset(
        repo_id="lerobot/aloha_sim_insertion_human_image",
        episodes=[0],
        download_videos=False,  # skip downloading video files entirely
    )
    item = dataset[0]
    # item["observation.images.top"] will be a zero tensor with the correct shape

    # Works with DataLoader as expected:
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    for batch in loader:
        # batch["observation.images.top"] → fake zero frames (fast)
        # batch["observation.state"], batch["action"], etc. → real data
        pass
"""

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PatchedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that skips video decoding and returns fake frames.

    __getitem__ flow:
      1. hf_dataset[idx]          → parquet read (state/action, fast)
      2. _query_hf_dataset()      → already skips video_keys (fast)
      3. _get_query_timestamps()  → lightweight timestamp lookup (fast)
      4. _query_videos()          → THIS is the bottleneck (decode_video_frames)
                                    we override this to return zero tensors

    By also passing download_videos=False to __init__, you skip downloading
    the mp4 files entirely.
    """

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # shape from metadata: (channels, height, width)
            shape = tuple(self.meta.features[vid_key]["shape"])
            num_frames = len(query_ts)

            if num_frames == 1:
                # Single frame → (C, H, W), matching original squeeze(0) behavior
                item[vid_key] = torch.zeros(shape, dtype=torch.float32)
            else:
                # Multi-frame → (num_frames, C, H, W)
                item[vid_key] = torch.zeros((num_frames, *shape), dtype=torch.float32)

        return item
