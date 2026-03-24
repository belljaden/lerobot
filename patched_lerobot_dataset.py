"""
PatchedLeRobotDataset: LeRobotDataset subclass that selectively skips
video decoding based on keyword matching.

Usage:
    from patched_lerobot_dataset import PatchedLeRobotDataset

    dataset = PatchedLeRobotDataset(
        repo_id="lerobot/aloha_sim_insertion_human_image",
        episodes=[0],
        skip_video_keywords=["wrist", "side"],
    )

    # "observation.images.wrist"      → zero tensor (skipped)
    # "observation.images.left_wrist" → zero tensor (skipped)
    # "observation.images.top"        → real decoded frame
"""

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PatchedLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, skip_video_keywords: list[str] | None = None, **kwargs):
        self._skip_video_keywords = skip_video_keywords or []
        super().__init__(*args, **kwargs)

    def _should_skip(self, vid_key: str) -> bool:
        return any(kw in vid_key for kw in self._skip_video_keywords)

    def _query_videos(self, query_timestamps, ep_idx):
        skipped = {}
        kept = {}
        for vid_key, ts in query_timestamps.items():
            if self._should_skip(vid_key):
                shape = tuple(self.meta.features[vid_key]["shape"])
                n = len(ts)
                skipped[vid_key] = torch.zeros(shape if n == 1 else (n, *shape), dtype=torch.float32)
            else:
                kept[vid_key] = ts

        item = skipped
        if kept:
            item.update(super()._query_videos(kept, ep_idx))
        return item
