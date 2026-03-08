"""Tiện ích metric cho ttswd — tái xuất từ _common._metric_base."""
from _common._metric_base import (
    _pairwise_weighted_euclidean,
    _streaming_diameter,
    _approx_diameter,
    _centroid_diameter_upper_bound,
    _bbox_diameter_upper_bound,
    _RowLRUCache,
    _dist_row_streaming,
    _auto_select_mode,
)

__all__ = [
    "_pairwise_weighted_euclidean",
    "_streaming_diameter",
    "_approx_diameter",
    "_centroid_diameter_upper_bound",
    "_bbox_diameter_upper_bound",
    "_RowLRUCache",
    "_dist_row_streaming",
    "_auto_select_mode",
]
