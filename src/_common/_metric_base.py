"""Tiện ích metric dùng chung: khoảng cách Euclid có trọng số, ước lượng đường kính,
bộ nhớ đệm hàng streaming, và chọn chế độ tự động.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Khoảng cách Euclid có trọng số từng cặp
# ---------------------------------------------------------------------------

def _pairwise_weighted_euclidean(
    A: np.ndarray,
    B: np.ndarray,
    w: Optional[np.ndarray] = None,
    D_out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tính khoảng cách Euclid có trọng số; D_out là buffer cấp phát trước (tuỳ chọn)."""
    A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
    if w is not None:
        ws = np.sqrt(np.asarray(w, np.float32))
        A, B = A * ws, B * ws
    n, m = A.shape[0], B.shape[0]
    D = D_out if D_out is not None else np.zeros((n, m), np.float32)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    AA = np.sum(A * A, axis=1, keepdims=True)
    D_full = AA + BB - 2.0 * (A @ B.T)
    np.maximum(D_full, 0.0, out=D_full)
    np.sqrt(D_full, out=D)
    return D


# ---------------------------------------------------------------------------
# Ước lượng đường kính
# ---------------------------------------------------------------------------

def _streaming_diameter(coords_w: np.ndarray, block_size: int = 1024) -> float:
    """Tính khoảng cách cực đại từng cặp mà không cần cấp phát toàn bộ ma trận."""
    X = np.asarray(coords_w, np.float32)
    n = X.shape[0]
    if n == 0:
        return 0.0
    block = max(int(block_size), 1)
    diam = 0.0
    for i in range(0, n, block):
        A = X[i : i + block]
        for j in range(i, n, block):
            B = X[j : j + block]
            D_blk = _pairwise_weighted_euclidean(A, B)
            diam = max(diam, float(np.max(D_blk)))
    return diam


def _approx_diameter(
    coords_w: np.ndarray,
    n_pivots: int = 5,
    block_size: int = 1024,
    random_state=None,
) -> float:
    """Xấp xỉ đường kính bằng double sweep ngẫu nhiên.

    Với mỗi điểm xuất phát ngẫu nhiên:
      1. Tìm u = điểm xa nhất từ điểm xuất phát  (O(n))
      2. Tìm v = điểm xa nhất từ u               (O(n))
      3. diam_approx = d(u, v)
    Trả về max qua n_pivots lần thử.

    Độ phức tạp: O(n_pivots × n), nhanh hơn nhiều so với _streaming_diameter O(n²).
    Đảm bảo: kết quả ≥ diam / 2.
    """
    n = np.asarray(coords_w).shape[0]
    if n <= 1:
        return 0.0
    rng = np.random.default_rng(random_state)
    buf = np.empty(n, np.float32)
    best = 0.0
    for s in rng.integers(0, n, size=n_pivots):
        _dist_row_streaming(coords_w, int(s), out=buf, block_size=block_size)
        u = int(np.argmax(buf))
        _dist_row_streaming(coords_w, u, out=buf, block_size=block_size)
        v_dist = float(np.max(buf))
        if v_dist > best:
            best = v_dist
    return best


# ---------------------------------------------------------------------------
# Cache LRU theo hàng và tính hàng khoảng cách dạng streaming
# ---------------------------------------------------------------------------

class _RowLRUCache:
    """Cache LRU cho các hàng khoảng cách (pivot_idx → np.ndarray).

    Dùng OrderedDict để get/put O(1) thay vì list.remove() O(n).
    """

    def __init__(self, max_entries: int = 16):
        self.max_entries = max_entries
        self._store: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: int, val: np.ndarray) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_entries:
                self._store.popitem(last=False)  # đẩy phần tử ít dùng nhất ra
        self._store[key] = val


def _dist_row_streaming(
    coords_w: np.ndarray,
    pivot_idx: int,
    out: Optional[np.ndarray] = None,
    block_size: int = 1024,
) -> np.ndarray:
    """Tính khoảng cách từ một điểm pivot đến tất cả điểm khác (tiết kiệm bộ nhớ)."""
    X = np.asarray(coords_w, np.float32)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, np.float32)
    buf = out if out is not None else np.empty(n, np.float32)
    pivot = X[pivot_idx]
    if n <= 50000:
        diff = X - pivot
        np.sqrt(np.sum(diff * diff, axis=1), out=buf)
        return buf
    block = max(int(block_size), 1)
    pos = 0
    pivot_2d = pivot.reshape(1, -1)
    for start in range(0, n, block):
        seg = X[start : start + block]
        dists = np.sqrt(np.sum((seg - pivot_2d) ** 2, axis=1))
        m = dists.shape[0]
        buf[pos : pos + m] = dists
        pos += m
    return buf


# ---------------------------------------------------------------------------
# Chọn chế độ tính khoảng cách
# ---------------------------------------------------------------------------

def _auto_select_mode(
    n_points: int,
    distance_mode: str = "auto",
    precompute_threshold: int = 10000,
) -> str:
    """Chọn 'precompute' hoặc 'streaming' dựa trên kích thước tập dữ liệu.

    'precompute': cấp phát toàn bộ ma trận N×N một lần (truy cập ngẫu nhiên nhanh).
    'streaming':  tính từng hàng theo yêu cầu với cache LRU — phù hợp với N lớn.
    """
    if distance_mode != "auto":
        return distance_mode.lower()
    return "precompute" if n_points <= precompute_threshold else "streaming"
