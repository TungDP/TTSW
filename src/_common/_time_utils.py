"""Tiện ích chỉ số thời gian và ghép chuỗi dùng chung."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def add_time_index(series: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Thêm cột thời gian chuẩn hoá [0, 1] vào mảng chuỗi thời gian (T, d).

    Tham số
    ----------
    series    : mảng đặc trưng (T, d)
    normalize : nếu True thì cột thời gian chạy 0 → 1; ngược lại 0 → T-1

    Kết quả
    -------
    Mảng (T, d+1) với thời gian được thêm vào cột 0
    """
    S = np.asarray(series, dtype=float)
    m = S.shape[0]
    if m == 0:
        return np.zeros((0, S.shape[1] + 1))
    t = np.arange(m, dtype=float)
    if normalize and m > 1:
        t /= (m - 1.0)
    return np.hstack([t.reshape(-1, 1), S])


def stack_union_points_with_time(
    X_train_series: List[np.ndarray],
    X_test_series: List[np.ndarray],
    normalize_time: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Ghép tất cả chuỗi train + test (có chỉ số thời gian cục bộ) thành một mảng.

    Mỗi chuỗi được thêm cột thời gian chuẩn hoá trước khi ghép, để tất cả
    các điểm dùng chung một không gian toạ độ.

    Tham số
    ----------
    X_train_series : danh sách mảng (T_i, d) — chuỗi train
    X_test_series  : danh sách mảng (T_j, d) — chuỗi test
    normalize_time : nếu True thì thời gian chạy 0 → 1 theo từng chuỗi

    Kết quả
    -------
    coords_all     : (N, d+1) tập hợp tất cả điểm (train trước, test sau)
    seq_membership : (N,) mảng int — id chuỗi của từng điểm (đánh số từ 0)
    seq_boundaries : danh sách cặp (start, end) cho từng chuỗi,
                     theo thứ tự X_train_series + X_test_series
    """
    all_chunks: List[np.ndarray] = []
    seq_membership: List[int] = []
    seq_boundaries: List[Tuple[int, int]] = []

    all_series = list(X_train_series) + list(X_test_series)
    offset = 0
    for sid, s in enumerate(all_series):
        arr = add_time_index(s, normalize=normalize_time)
        all_chunks.append(arr)
        seq_membership.extend([sid] * len(arr))
        seq_boundaries.append((offset, offset + len(arr)))
        offset += len(arr)

    coords_all = np.vstack(all_chunks) if all_chunks else np.zeros((0, 0))
    return coords_all, np.array(seq_membership, np.int32), seq_boundaries
