"""Xây dựng forest TTSWD toàn cục và metadata."""
from __future__ import annotations

import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ._metric import (
    _auto_select_mode,
    _pairwise_weighted_euclidean,
    _streaming_diameter,
)

_APPROX_DIAM_THRESHOLD = 400_000  # dùng xấp xỉ khi N > ngưỡng này
from ._tree import TTSWDTree, _SpatialIndex, _ttswd_tree_unified


class GlobalMeta:
    """Metadata của forest TTSWD toàn cục và ánh xạ chuỗi."""

    def __init__(
        self,
        time_weight: float,
        wdim: np.ndarray,
        time_col: int,
        feature_cols: List[int],
        w_avg: float,
        *,
        seq_boundaries: Optional[List[Tuple[int, int]]] = None,
        distance_mode: str = "auto",
        streaming_block_size: int = 1024,
    ):
        self.time_weight = float(time_weight)
        self.wdim = wdim
        self.time_col = int(time_col)
        self.feature_cols = list(feature_cols)
        self.w_avg = float(w_avg)
        self.seq_boundaries = seq_boundaries or []
        self.distance_mode = str(distance_mode)
        self.streaming_block_size = int(streaming_block_size)
        self.build_tree_sec = 0.0
        self.distance_calc_sec = 0.0
        self.build_modes: List[str] = []
        self.build_times_per_tree: List[float] = []
        self.actual_mode: str = "unknown"

    def get_mode_summary(self) -> str:
        if not self.build_modes:
            return "Không có thông tin chế độ"
        mode_counts = Counter(self.build_modes)
        total_time = sum(self.build_times_per_tree)
        return f"Chế độ: {dict(mode_counts)} | Tổng thời gian: {total_time:.4f}s"


def build_global_ttswd_forest(
    coords_all: np.ndarray,
    n_trees: int = 1,
    time_weight: float = 64.0,
    time_col: int = 0,
    feature_cols: Optional[List[int]] = None,
    random_state=None,
    alpha=None,
    distance_mode: str = "auto",
    streaming_block_size: int = 1024,
    distance_scale: float = 1.0,
    normalize_features: bool = True,
    approx_diameter: bool = False,
) -> Tuple[List[TTSWDTree], GlobalMeta]:
    """Xây nhiều cây TTSWD toàn cục từ toạ độ đã ghép.

    approx_diameter : nếu True, luôn dùng bounding-box diagonal (cận trên đảm bảo, O(n·d), 1 pass)
                      thay vì tính đường kính chính xác O(n²).
                      Mặc định False; tự động bật khi N > 400 000.

    Trả về (trees, meta).
    """
    X = np.asarray(coords_all, float)
    if X.ndim != 2:
        X = np.asarray(coords_all, float).reshape(-1, 1)
    N, d = X.shape
    if feature_cols is None:
        feature_cols = [j for j in range(d) if j != time_col]

    # Chuẩn hoá đặc trưng: chia mỗi cột đặc trưng cho độ lệch chuẩn để
    # tập dữ liệu biên độ lớn (vd. gia tốc thô) không làm phình đường kính cây
    # và tạo ra cây quá sâu (L lớn).
    if normalize_features and feature_cols:
        X = X.copy()
        feat_std = np.std(X[:, feature_cols], axis=0)
        feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)
        X[:, feature_cols] /= feat_std

    tw = float(time_weight)

    wdim = np.ones(d, float)
    if d > 0:
        wdim[time_col] = tw

    coords_weighted = X * np.sqrt(wdim) if d > 0 else X

    # Nhân tất cả khoảng cách theo distance_scale (dịch chuyển xác suất tách cặp
    # qua các tầng cây; 1.0 = không thay đổi).
    if distance_scale != 1.0:
        coords_weighted = coords_weighted * float(distance_scale)

    # Chọn chế độ và tính trước dữ liệu chia sẻ (chỉ đọc, dùng chung cho tất cả cây)
    distance_mode = (distance_mode or "auto").lower()
    selected_mode = _auto_select_mode(coords_weighted.shape[0], distance_mode, precompute_threshold=24444)

    shared_D = None
    shared_diameter = None
    shared_spatial_index = None
    if selected_mode == "precompute":
        shared_D = _pairwise_weighted_euclidean(coords_weighted, coords_weighted)
    elif selected_mode == "streaming":
        # Xây chỉ mục không gian trước — cKDTree lưu sẵn bounding box toàn cục
        # trong quá trình xây, cho phép lấy diameter_upper_bound() miễn phí.
        shared_spatial_index = _SpatialIndex(coords_weighted)
        # Tính đường kính một lần, chia sẻ cho tất cả cây.
        if approx_diameter or N > _APPROX_DIAM_THRESHOLD:
            # Tận dụng metadata của spatial index: O(d) với cKDTree, O(n·d) với BallTree.
            shared_diameter = shared_spatial_index.diameter_upper_bound()
        else:
            shared_diameter = _streaming_diameter(coords_weighted, block_size=streaming_block_size)

    base_seed = None if random_state is None else int(random_state)
    seeds = [None if base_seed is None else base_seed + t for t in range(n_trees)]

    def _build_one(rs):
        return _ttswd_tree_unified(
            coords_weighted,
            distance_mode=distance_mode,
            random_state=rs,
            alpha=alpha,
            block_size=streaming_block_size,
            precomputed_D=shared_D,               # mảng numpy chỉ đọc → an toàn đa luồng
            precomputed_diameter=shared_diameter,
            shared_spatial_index=shared_spatial_index,  # xây một lần, dùng chung
        )

    # Xây cây song song khi có nhiều cây.
    # shared_D / coords_weighted là mảng numpy chỉ đọc: an toàn cho threading.
    n_jobs_tree = min(os.cpu_count() or 1, n_trees, 8)
    if n_trees > 1 and n_jobs_tree > 1:
        results = list(tqdm(
            Parallel(n_jobs=n_jobs_tree, backend="threading", return_as="generator")(
                delayed(_build_one)(rs) for rs in seeds
            ),
            total=n_trees, desc="TTSWD trees",
        ))
    else:
        results = [_build_one(rs) for rs in tqdm(seeds, desc="TTSWD trees")]

    trees = [r[0] for r in results]
    build_modes = [r[1]["mode_used"] for r in results]
    build_times = [r[1]["build_time"] for r in results]

    ws = [np.mean(t.weight[1:]) if t.weight.size > 1 else 0.0 for t in trees]
    meta = GlobalMeta(
        tw,
        wdim,
        time_col,
        feature_cols,
        float(np.mean(ws)) if ws else 0.0,
        seq_boundaries=None,
        distance_mode=distance_mode,
        streaming_block_size=streaming_block_size,
    )
    meta.build_modes = build_modes
    meta.build_times_per_tree = build_times
    meta.actual_mode = build_modes[0] if build_modes else "unknown"

    return trees, meta
