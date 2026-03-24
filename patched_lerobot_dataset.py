"""
PatchedLeRobotDataset: LeRobotDataset subclass that completely excludes
certain video keys from loading based on keyword matching.

Usage:
    from patched_lerobot_dataset import PatchedLeRobotDataset

    dataset = PatchedLeRobotDataset(
        repo_id="lerobot/aloha_sim_insertion_human_image",
        episodes=[0],
        skip_video_keywords=["wrist", "side"],
    )

    # "observation.images.wrist"      → not in item at all (no memory usage)
    # "observation.images.left_wrist" → not in item at all
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

    def _get_query_timestamps(self, current_ts, query_indices=None):
        # Get all timestamps, then drop skipped keys
        query_timestamps = super()._get_query_timestamps(current_ts, query_indices)
        return {k: v for k, v in query_timestamps.items() if not self._should_skip(k)}

    def _query_videos(self, query_timestamps, ep_idx):
        # query_timestamps already has skipped keys removed by _get_query_timestamps
        if not query_timestamps:
            return {}
        return super()._query_videos(query_timestamps, ep_idx)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Remove skipped keys from the final item
        return {k: v for k, v in item.items() if not self._should_skip(k)}
