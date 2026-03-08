"""Tính khoảng cách Tree-Wasserstein (TW) thưa cho TTSWD.

Khoảng cách TW giữa hai phân phối A, B trên cây T:
  TW(A, B) = Σ_{v ∈ V(T)} weight(v) · |m_A(v) - m_B(v)|

trong đó:
  weight(v) = 2^level(v)   — trọng số cạnh node nội tại (0 cho gốc)
  m_A(v)    = mass của phân phối A tại cây con gốc v
            = Σ_{p ∈ A : v nằm trên đường leaf(p)→root} prob(p)

Với phân phối đồng đều: m_A(v) = |{p ∈ A : v ∈ path(leaf(p)→root)}| / |A|

Tính hiệu quả:
  - Biểu diễn thưa: (nodes_sorted, mass_sorted) chỉ lưu node có mass > 0.
  - _tw_distance_sparse: merge-sort hai mảng thưa, O(k_A + k_B) thay vì O(N).
  - Forest: TW trung bình qua n_trees cây để giảm phương sai ước lượng.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ._tree import TTSWDTree

# ---------------------------------------------------------------------------
# Numba JIT: vòng lặp trong cùng tính TW distance giữa hai biểu diễn thưa
#
# Thuật toán merge-sort:
#   Duyệt đồng thời hai mảng đã sắp xếp (nodes_a, mass_a) và (nodes_b, mass_b).
#   Tại mỗi bước, lấy node có id nhỏ hơn (hoặc cả hai nếu bằng nhau):
#     - Chỉ trong a: diff = +mass_a[i]        (m_B(v) = 0)
#     - Chỉ trong b: diff = -mass_b[j]        (m_A(v) = 0)
#     - Trong cả hai: diff = mass_a[i] - mass_b[j]
#   Cộng |diff| × weight[nid] vào tổng chi phí.
#
# Độ phức tạp: O(k_A + k_B) với k_A, k_B = số node thưa của A, B.
# Fallback sang Python thuần nếu numba chưa cài.
# ---------------------------------------------------------------------------
try:
    import numba as _nb

    @_nb.njit(cache=True, fastmath=True)
    def _tw_distance_sparse(node_weight, nodes_a, mass_a, nodes_b, mass_b):
        """Tính TW distance giữa hai biểu diễn mass thưa.

        Tham số
        -------
        node_weight : float64[N]   — weight[v] = 2^level(v) cho mọi node v
        nodes_a     : int32[k_A]   — id node của A (đã sắp xếp tăng dần)
        mass_a      : float64[k_A] — mass tương ứng tại các node của A
        nodes_b     : int32[k_B]   — id node của B (đã sắp xếp tăng dần)
        mass_b      : float64[k_B] — mass tương ứng tại các node của B

        Trả về
        ------
        cost : float64 — TW(A, B) = Σ_v weight(v) · |m_A(v) - m_B(v)|
        """
        i = 0
        j = 0
        na = nodes_a.shape[0]
        nb = nodes_b.shape[0]
        cost = 0.0
        while i < na or j < nb:
            if j >= nb or (i < na and nodes_a[i] < nodes_b[j]):
                # Node chỉ có trong A: m_B(v) = 0 → |diff| = mass_a[i]
                nid = nodes_a[i]
                diff = mass_a[i]
                i += 1
            elif i >= na or nodes_b[j] < nodes_a[i]:
                # Node chỉ có trong B: m_A(v) = 0 → |diff| = mass_b[j]
                nid = nodes_b[j]
                diff = -mass_b[j]
                j += 1
            else:
                # Node trong cả hai: diff = m_A(v) - m_B(v)
                nid = nodes_a[i]
                diff = mass_a[i] - mass_b[j]
                i += 1
                j += 1
            cost += abs(diff) * node_weight[nid]
        return cost

except ImportError:

    def _tw_distance_sparse(node_weight, nodes_a, mass_a, nodes_b, mass_b):
        """Tính TW distance giữa hai biểu diễn mass thưa (Python fallback)."""
        i = 0
        j = 0
        na = nodes_a.size
        nb = nodes_b.size
        cost = 0.0
        w = node_weight
        while i < na or j < nb:
            if j >= nb or (i < na and nodes_a[i] < nodes_b[j]):
                nid = nodes_a[i]
                diff = mass_a[i]
                i += 1
            elif i >= na or nodes_b[j] < nodes_a[i]:
                nid = nodes_b[j]
                diff = -mass_b[j]
                j += 1
            else:
                nid = nodes_a[i]
                diff = mass_a[i] - mass_b[j]
                i += 1
                j += 1
            cost += abs(float(diff)) * float(w[nid])
        return cost


def _series_indices_from_boundaries(
    seq_boundaries: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """Chuyển danh sách (start, end) thành danh sách mảng chỉ số điểm.

    seq_boundaries[s] = (start_s, end_s): chuỗi s chiếm các điểm
    [start_s, end_s) trong đám mây điểm toàn cục.
    """
    indices: List[np.ndarray] = []
    for start, end in seq_boundaries:
        if end <= start:
            indices.append(np.zeros(0, np.int32))
        else:
            indices.append(np.arange(start, end, dtype=np.int32))
    return indices


def _build_series_masses(
    trees: List[TTSWDTree],
    series_indices: List[np.ndarray],
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """Xây biểu diễn mass thưa cho tất cả chuỗi trên tất cả cây.

    Kết quả: masses_per_tree[t][s] = (nodes_sorted, mass_sorted)
    là biểu diễn thưa của chuỗi s trên cây t.
    Được tính trước một lần để tái sử dụng cho tất cả cặp (i, j).
    """
    masses_per_tree: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    for tree in trees:
        per_series: List[Tuple[np.ndarray, np.ndarray]] = []
        for idx in series_indices:
            if idx.size == 0:
                per_series.append((np.zeros(0, np.int32), np.zeros(0, np.float64)))
                continue
            # Lấy id lá của các điểm thuộc chuỗi này
            leaf_ids = tree.leaf_of_point[idx]
            # Xây mass thưa: duyệt đường lá→gốc, tích luỹ 1/|S|
            nodes, mass = tree.build_series_mass_uniform(leaf_ids)
            per_series.append((nodes, mass))
        masses_per_tree.append(per_series)
    return masses_per_tree


def compute_distance_matrices_ttswd(
    trees: List[TTSWDTree],
    seq_boundaries: List[Tuple[int, int]],
    n_tr: int,
    n_te: int,
    desc_prefix: str = "TTSWD",
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính ma trận khoảng cách TW từng cặp dùng biểu diễn mass thưa.

    Khoảng cách giữa chuỗi i và j = trung bình TW(i,j) qua n_trees cây:
      D[i,j] = (1/n_trees) · Σ_t TW_t(i, j)

    Tính trung bình qua nhiều cây (forest) giảm phương sai nhờ trung bình hoá
    các cây ngẫu nhiên độc lập (mỗi cây có β, σ khác nhau).

    Tham số
    -------
    trees          : danh sách TTSWDTree đã xây
    seq_boundaries : [(start, end)] — ranh giới từng chuỗi trong đám mây điểm
    n_tr           : số chuỗi train (chuỗi 0..n_tr-1)
    n_te           : số chuỗi test (chuỗi n_tr..n_tr+n_te-1)
    desc_prefix    : tiền tố cho thanh tiến trình

    Trả về
    ------
    D_tr : float64[n_tr, n_tr] — ma trận khoảng cách train–train (đối xứng)
    D_te : float64[n_te, n_tr] — ma trận khoảng cách test–train
    """
    n_trees = len(trees)
    if n_trees == 0:
        return np.zeros((n_tr, n_tr), np.float64), np.zeros((n_te, n_tr), np.float64)

    # --- Tính trước mass thưa cho tất cả chuỗi trên tất cả cây ---
    # masses_per_tree[t][s] = (nodes_sorted, mass_sorted) — O(n_trees × n_series × L)
    series_indices = _series_indices_from_boundaries(seq_boundaries)
    masses_per_tree = _build_series_masses(trees, series_indices)
    tree_weights = [t.weight for t in trees]
    inv_n_trees = 1.0 / n_trees

    def _tw_pair(i: int, j: int) -> float:
        """TW distance trung bình qua tất cả cây giữa chuỗi i và j."""
        total = 0.0
        for t_idx in range(n_trees):
            nodes_i, mass_i = masses_per_tree[t_idx][i]
            nodes_j, mass_j = masses_per_tree[t_idx][j]
            total += _tw_distance_sparse(tree_weights[t_idx], nodes_i, mass_i, nodes_j, mass_j)
        return total * inv_n_trees

    # --- Ma trận train–train: D_tr[i,j] = D_tr[j,i] (đối xứng) ---
    # Chỉ tính n_tr×(n_tr-1)/2 cặp phía trên đường chéo
    D_tr = np.zeros((n_tr, n_tr), np.float64)

    def _compute_row_tr(i: int) -> List[Tuple[int, int, float]]:
        results = []
        for j in range(i + 1, n_tr):
            results.append((i, j, _tw_pair(i, j)))
        return results

    n_jobs = min(os.cpu_count() or 1, 8)
    if n_tr > 50 and n_jobs > 1:
        row_results = list(tqdm(
            Parallel(n_jobs=n_jobs, backend="threading", return_as="generator")(
                delayed(_compute_row_tr)(i) for i in range(n_tr)
            ),
            total=n_tr, desc=f"{desc_prefix} D_tr",
        ))
        for results in row_results:
            for i, j, val in results:
                D_tr[i, j] = D_tr[j, i] = val
    else:
        n_pairs_tr = n_tr * (n_tr - 1) // 2
        with tqdm(total=n_pairs_tr, desc=f"{desc_prefix} D_tr") as pbar:
            for i in range(n_tr):
                for j in range(i + 1, n_tr):
                    D_tr[i, j] = D_tr[j, i] = _tw_pair(i, j)
                    pbar.update(1)

    # --- Ma trận test–train: D_te[p,j] = TW(test_p, train_j) ---
    # Chuỗi test có id n_tr + p trong danh sách series_indices
    D_te = np.zeros((n_te, n_tr), np.float64)

    def _compute_row_te(p: int) -> List[Tuple[int, float]]:
        results = []
        sid_p = n_tr + p   # id chuỗi test p trong danh sách toàn cục
        for j in range(n_tr):
            results.append((j, _tw_pair(sid_p, j)))
        return results

    if n_te > 20 and n_jobs > 1:
        row_results = list(tqdm(
            Parallel(n_jobs=n_jobs, backend="threading", return_as="generator")(
                delayed(_compute_row_te)(p) for p in range(n_te)
            ),
            total=n_te, desc=f"{desc_prefix} D_te",
        ))
        for p, results in enumerate(row_results):
            for j, val in results:
                D_te[p, j] = val
    else:
        n_pairs_te = n_te * n_tr
        with tqdm(total=n_pairs_te, desc=f"{desc_prefix} D_te") as pbar:
            for p in range(n_te):
                sid_p = n_tr + p
                for j in range(n_tr):
                    D_te[p, j] = _tw_pair(sid_p, j)
                    pbar.update(1)

    return D_tr, D_te
