"""
Monkey-patch to selectively skip video decoding in LeRobotDataset.

Usage:
    import patched_lerobot_dataset  # just import to apply patch

    # Then use LeRobotDataset as normal — matched video keys will return zero tensors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id="...", episodes=[0])

Customize SKIP_VIDEO_KEYWORDS to control which video keys are skipped.
A vid_key is skipped if it contains ANY of the keywords (substring match).

Examples:
    SKIP_VIDEO_KEYWORDS = ["wrist"]
      → skips "observation.images.wrist", "observation.images.left_wrist", etc.

    SKIP_VIDEO_KEYWORDS = ["wrist", "side"]
      → skips any vid_key containing "wrist" OR "side"

    SKIP_VIDEO_KEYWORDS = []
      → skip nothing (no-op)
"""

import torch
import lerobot.datasets.lerobot_dataset as _ld_module

# ===== Configure here =====
SKIP_VIDEO_KEYWORDS = ["wrist", "side"]
# ===========================

_original_query_videos = _ld_module.LeRobotDataset._query_videos


def _should_skip(vid_key: str) -> bool:
    return any(kw in vid_key for kw in SKIP_VIDEO_KEYWORDS)


def _patched_query_videos(self, query_timestamps, ep_idx):
    skipped = {}
    kept = {}
    for vid_key, ts in query_timestamps.items():
        if _should_skip(vid_key):
            shape = tuple(self.meta.features[vid_key]["shape"])
            n = len(ts)
            skipped[vid_key] = torch.zeros(shape if n == 1 else (n, *shape), dtype=torch.float32)
        else:
            kept[vid_key] = ts

    item = skipped
    if kept:
        item.update(_original_query_videos(self, kept, ep_idx))
    return item


_ld_module.LeRobotDataset._query_videos = _patched_query_videos
