"""
ttswd — Khoảng cách Wasserstein trên cây cho chuỗi thời gian,
sử dụng biểu diễn mass cây con thưa.

API công khai:
- stack_union_points_with_time
- build_global_ttswd_forest
- compute_distance_matrices_ttswd
- GlobalMeta
- TTSWDTree
"""
from __future__ import annotations

from ._forest import GlobalMeta, build_global_ttswd_forest
from ._time_utils import stack_union_points_with_time
from ._tree import TTSWDTree
from ._wasserstein import compute_distance_matrices_ttswd

__all__ = [
    "stack_union_points_with_time",
    "build_global_ttswd_forest",
    "compute_distance_matrices_ttswd",
    "GlobalMeta",
    "TTSWDTree",
]
