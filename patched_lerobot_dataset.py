"""
PatchedLeRobotDataset: LeRobotDataset subclass that skips video decoding
and returns fake (zero) frames matching the expected video shape.

Usage:
    from patched_lerobot_dataset import PatchedLeRobotDataset

    dataset = PatchedLeRobotDataset(
        repo_id="lerobot/aloha_sim_insertion_human_image",
        episodes=[0],
    )
    item = dataset[0]
    # item["observation.images.top"] will be a zero tensor with the correct shape
"""

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PatchedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that returns fake video frames instead of decoding actual videos.

    This is useful when you want to work with the non-video data (states, actions, etc.)
    without needing actual video files or paying the cost of video decoding.

    Fake frames are zero-valued tensors shaped according to the feature metadata,
    e.g. shape (channels, height, width) as defined in info.json.
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
                # Single frame query → return shape (C, H, W) to match squeeze(0) behavior
                item[vid_key] = torch.zeros(shape, dtype=torch.float32)
            else:
                # Multi-frame query → return shape (num_frames, C, H, W)
                item[vid_key] = torch.zeros((num_frames, *shape), dtype=torch.float32)

        return item
