"""Xây dựng cây 2-HST của TTSWD: lớp TTSWDTree và các hàm xây cây.

Thuật toán cốt lõi: FRT (Fakcharoenphol–Rao–Talwar 2004)
---------------------------------------------------------
Cho n điểm với metric D, FRT xây một cây ngẫu nhiên T sao cho:
  - Với mọi cặp (i,j): d_T(i,j) ≥ D(i,j)  (cây không co khoảng cách)
  - E[d_T(i,j)] ≤ O(log n) · D(i,j)         (kỳ vọng giãn nở tối đa O(log n))

Cây kết quả là 2-HST (2-Hierarchically Separated Tree):
  - Mỗi node nội tại ở tầng i có trọng số cạnh (lên cha) = 2^i
  - Gốc ở tầng L = ⌈log₂(diam)⌉, trọng số cạnh = 0
  - Lá không có trọng số cạnh riêng (chứa điểm gốc)

Khoảng cách cây giữa hai điểm = tổng trọng số các cạnh trên đường nối chúng.
Đây là thước đo hợp lệ cho Tree-Wasserstein (TW) distance.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree as _cKDTree
    _HAS_KDTREE = True
except ImportError:  # pragma: no cover
    _cKDTree = None  # type: ignore[assignment]
    _HAS_KDTREE = False

try:
    from sklearn.neighbors import BallTree as _BallTree
    _HAS_BALLTREE = True
except ImportError:  # pragma: no cover
    _BallTree = None  # type: ignore[assignment]
    _HAS_BALLTREE = False

# d ≤ ngưỡng này → cKDTree (O(log n + k) trong không gian Euclid thấp chiều)
# d >  ngưỡng này → BallTree (cắt tỉa bóng–bóng hiệu quả hơn cho d lớn)
_KDTREE_MAX_DIM = 5


class _SpatialIndex:
    """Tự chọn cKDTree (d ≤ 5) hoặc BallTree (d > 5) cho truy vấn bóng chính xác.

    cKDTree  dùng chia siêu phẳng → O(log n + k) chính xác cho d ≤ ~5.
    BallTree dùng chia siêu cầu   → cắt tỉa chặt hơn cho truy vấn bán kính d > 5.
    Cả hai đều trả kết quả chính xác (không xấp xỉ).
    """

    __slots__ = ("_tree", "_use_kd", "_X")

    def __init__(self, X: np.ndarray) -> None:
        self._X = np.asarray(X, np.float32)
        d = X.shape[1]
        use_kd = d <= _KDTREE_MAX_DIM and _HAS_KDTREE
        if use_kd:
            self._tree = _cKDTree(X)
            self._use_kd = True
        elif _HAS_BALLTREE:
            self._tree = _BallTree(X)
            self._use_kd = False
        elif _HAS_KDTREE:                    # BallTree chưa cài: dùng cKDTree thay thế
            self._tree = _cKDTree(X)
            self._use_kd = True
        else:
            raise RuntimeError(
                "Không có chỉ mục không gian: hãy cài scipy (cKDTree) "
                "hoặc scikit-learn (BallTree)."
            )

    def diameter_upper_bound(self) -> float:
        """Cận trên đường kính lấy từ metadata của spatial index.

        cKDTree : tree.mins / tree.maxes (bounding box toàn cục) được lưu
                  sẵn trong quá trình xây cây → hoàn toàn miễn phí.
        BallTree: không expose bbox trực tiếp → tính bbox 1 pass O(n·d).

        Kết quả = ||max(X) − min(X)||₂ ≥ diam_true (đảm bảo lý thuyết FRT).
        """
        if self._use_kd:
            lo = np.asarray(self._tree.mins, np.float32)
            hi = np.asarray(self._tree.maxes, np.float32)
        else:
            lo = self._X.min(axis=0)
            hi = self._X.max(axis=0)
        return float(np.sqrt(np.sum((hi - lo) ** 2)))

    def query_ball(self, point: np.ndarray, r: float):
        """Trả về chỉ số tất cả điểm trong bán kính r (chính xác)."""
        if self._use_kd:
            return self._tree.query_ball_point(point, r=r)
        return self._tree.query_radius(point.reshape(1, -1), r=r)[0]


from ._metric import (
    _auto_select_mode,
    _pairwise_weighted_euclidean,
    _streaming_diameter,
)

# ---------------------------------------------------------------------------
# Numba JIT: tích luỹ mass theo đường lá → gốc
#
# Mỗi chuỗi thời gian S được biểu diễn bằng phân phối xác suất đồng đều
# trên tập lá {leaf(p) : p ∈ S}.  Mass của node v = tổng xác suất của
# các lá trong cây con gốc v, tức là:
#
#   m_S(v) = |{p ∈ S : v nằm trên đường leaf(p) → root}| / |S|
#
# Thuật toán: với mỗi lá leaf(p), đi từ leaf(p) lên gốc và cộng 1/|S|
# vào acc[v] tại mỗi node v trên đường đi.
#
# Kết quả: mảng thưa (nodes_sorted, mass_sorted) chỉ chứa node có mass > 0.
# Fallback sang Python thuần nếu numba chưa cài.
# ---------------------------------------------------------------------------
try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _fill_mass_uniform(leaf_ids, parent, mark, acc, visit, w, nodes_buf):
        """Duyệt đường lá→gốc, tích luỹ mass đồng đều (w = 1/|S|).

        Tham số
        -------
        leaf_ids  : int32[m]   — chỉ số node lá của từng điểm trong chuỗi
        parent    : int32[N]   — parent[v] = cha của node v; -1 nếu là gốc
        mark      : int32[N]   — nhãn lần thăm để tránh reset acc giữa các lần gọi
        acc       : float64[N] — bộ tích luỹ mass (được cập nhật tại chỗ)
        visit     : int32      — id lần thăm hiện tại (tăng dần, tránh fill(0))
        w         : float64    — trọng số mỗi điểm = 1/|S|
        nodes_buf : int32[N]   — buffer để ghi lại node đã thăm (tránh alloc)

        Trả về
        ------
        count : int — số node đã được tích luỹ (để slice nodes_buf[:count])
        """
        count = 0
        for li in range(leaf_ids.shape[0]):
            v = leaf_ids[li]
            while v >= 0:
                if mark[v] != visit:
                    # Lần đầu thăm node v trong lần gọi này: khởi tạo acc[v]
                    mark[v] = visit
                    acc[v] = 0.0
                    nodes_buf[count] = v
                    count += 1
                acc[v] += w
                v = parent[v]
        return count

    @_nb.njit(cache=True)
    def _fill_mass_weighted(leaf_ids, leaf_weights, parent, mark, acc, visit, nodes_buf):
        """Duyệt đường lá→gốc, tích luỹ mass có trọng số tuỳ ý.

        Tương tự _fill_mass_uniform nhưng mỗi lá có trọng số riêng
        thay vì đồng đều 1/|S|.  Trả về số node đã thăm.
        """
        count = 0
        for li in range(leaf_ids.shape[0]):
            v = leaf_ids[li]
            if v < 0:
                continue
            w = leaf_weights[li]
            while v >= 0:
                if mark[v] != visit:
                    mark[v] = visit
                    acc[v] = 0.0
                    nodes_buf[count] = v
                    count += 1
                acc[v] += w
                v = parent[v]
        return count

except ImportError:

    def _fill_mass_uniform(leaf_ids, parent, mark, acc, visit, w, nodes_buf):
        count = 0
        for leaf in leaf_ids:
            v = int(leaf)
            while v >= 0:
                if mark[v] != visit:
                    mark[v] = visit
                    acc[v] = 0.0
                    nodes_buf[count] = v
                    count += 1
                acc[v] += w
                v = parent[v]
        return count

    def _fill_mass_weighted(leaf_ids, leaf_weights, parent, mark, acc, visit, nodes_buf):
        count = 0
        for li in range(len(leaf_ids)):
            v = int(leaf_ids[li])
            if v < 0:
                continue
            w = float(leaf_weights[li])
            while v >= 0:
                if mark[v] != visit:
                    mark[v] = visit
                    acc[v] = 0.0
                    nodes_buf[count] = v
                    count += 1
                acc[v] += w
                v = parent[v]
        return count


def _node_weights_from_levels(levels: np.ndarray, L: int) -> np.ndarray:
    """Tính trọng số cạnh theo tầng: node ở tầng i có weight = 2^i.

    Định nghĩa 2-HST:
      - Cạnh từ node nội tại ở tầng i lên cha có độ dài 2^i.
      - Gốc ở tầng L: trọng số cạnh lên cha = 0 (gốc không có cha).
      - Lá (tầng -1): không có trọng số cạnh riêng; khoảng cách giữa
        hai điểm = tổng trọng số các node nội tại trên đường nối chúng.

    Trong TW distance: weight[v] = 2^level[v] chính là hệ số nhân cho
    |m_A(v) - m_B(v)| khi tính khoảng cách TW.
    """
    weights = np.zeros(levels.shape, dtype=np.float64)
    if levels.size == 0:
        return weights
    mask = (levels >= 0) & (levels <= L)
    if np.any(mask):
        weights[mask] = np.exp2(levels[mask].astype(np.float64))
    weights[0] = 0.0  # gốc không đóng góp vào khoảng cách TW
    return weights


class TTSWDTree:
    """Cây 2-HST gọn với hỗ trợ mass cây con thưa.

    Biểu diễn nội bộ
    ----------------
    parent[v]       : cha của node v; -1 nếu v là gốc (node 0)
    level[v]        : tầng của node v; gốc = L, lá = -1, nội tại = 0..L-1
    leaf_of_point[p]: id node lá tương ứng với điểm p trong đám mây điểm
    weight[v]       : 2^level[v] cho node nội tại; 0 cho gốc và lá

    Thứ tự node: 0 = gốc, 1..nc-n-1 = node nội tại, nc-n..nc-1 = lá.
    """

    __slots__ = (
        "parent",
        "level",
        "leaf_of_point",
        "weight",
        "n_nodes",
        "n_points",
        "L",
        "_mark",
        "_acc",
        "_visit_id",
        "_nodes_buf",
    )

    def __init__(
        self,
        parent: np.ndarray,
        level: np.ndarray,
        leaf_of_point: np.ndarray,
        *,
        L: int,
    ):
        self.parent = np.asarray(parent, np.int32)
        self.level = np.asarray(level, np.int16)
        self.leaf_of_point = np.asarray(leaf_of_point, np.int32)
        self.n_nodes = int(self.parent.size)
        self.n_points = int(self.leaf_of_point.size)
        self.L = int(L)
        self.weight = _node_weights_from_levels(self.level, self.L)

        # Buffer dùng chung cho build_series_mass_* — tránh cấp phát lại mỗi lần gọi
        self._mark = np.zeros(self.n_nodes, dtype=np.int32)
        self._acc = np.zeros(self.n_nodes, dtype=np.float64)
        self._visit_id = 0
        self._nodes_buf = np.empty(self.n_nodes, dtype=np.int32)

    def _next_visit(self) -> int:
        """Tăng visit_id; reset mark nếu tràn int32."""
        self._visit_id += 1
        if self._visit_id >= np.iinfo(np.int32).max:
            self._mark.fill(0)
            self._visit_id = 1
        return self._visit_id

    def build_series_mass_uniform(
        self, leaf_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Xây biểu diễn mass thưa cho chuỗi với phân phối đồng đều.

        Mỗi điểm p trong chuỗi S đóng góp mass 1/|S| vào tất cả node
        trên đường leaf(p) → gốc.

        Trả về
        ------
        nodes_sorted : int32[k]   — id các node có mass > 0, sắp xếp tăng dần
        mass_sorted  : float64[k] — mass tương ứng tại các node đó

        Dạng thưa giúp _tw_distance_sparse duyệt bằng merge-sort O(k1 + k2)
        thay vì O(N) với N = tổng số node.
        """
        if leaf_ids.size == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float64)

        visit = np.int32(self._next_visit())
        w = 1.0 / float(leaf_ids.size)  # trọng số đồng đều mỗi điểm

        count = _fill_mass_uniform(
            leaf_ids.astype(np.int32, copy=False),
            self.parent, self._mark, self._acc,
            visit, w,
            self._nodes_buf,
        )
        if count == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float64)

        nodes_sorted = np.sort(self._nodes_buf[:count])
        mass_sorted = self._acc[nodes_sorted].copy()
        return nodes_sorted, mass_sorted

    def build_series_mass(
        self, leaf_ids: np.ndarray, leaf_weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Xây biểu diễn mass thưa với trọng số lá tuỳ ý (tổng phải = 1)."""
        if leaf_ids.size == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float64)

        visit = np.int32(self._next_visit())
        count = _fill_mass_weighted(
            leaf_ids.astype(np.int32, copy=False),
            np.asarray(leaf_weights, np.float64),
            self.parent, self._mark, self._acc,
            visit, self._nodes_buf,
        )
        if count == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float64)

        nodes_sorted = np.sort(self._nodes_buf[:count])
        mass_sorted = self._acc[nodes_sorted].copy()
        return nodes_sorted, mass_sorted


# ===========================================================================
# Thuật toán FRT: xây cây 2-HST ngẫu nhiên
# ===========================================================================
#
# Đầu vào: n điểm với metric D (ma trận khoảng cách hoặc toạ độ).
#
# Bước 1 — Tham số hoá ngẫu nhiên:
#   - Hoán vị ngẫu nhiên σ ∈ Sym(n): xác định thứ tự ưu tiên của các tâm
#     phân cụm tại mỗi tầng.
#   - Hệ số ngẫu nhiên β ~ Uniform[1, 2]: dịch chuyển ngẫu niên ranh giới
#     phân vùng để tránh điểm cố định bất lợi (key trick của FRT).
#
# Bước 2 — Độ cao cây:
#   - L = ⌈log₂(diam)⌉ với diam = max_{i,j} D(i,j).
#   - Đảm bảo bán kính cấp cao nhất (β·2^{L-1}) bao phủ toàn bộ đám mây.
#
# Bước 3 — Xây từng tầng i = L-1, L-2, ..., 0:
#   - Bán kính tầng i: r_i = β · 2^{i-1}
#     (Theo bài báo: r_i < 2^i để đảm bảo cạnh dài 2^i thoả điều kiện 2-HST)
#   - Duyệt các tâm theo thứ tự σ(0), σ(1), ...:
#       * Nếu σ(l) đã được gán trong tầng này: bỏ qua.
#       * Lấy B = {j : D(σ(l), j) ≤ r_i} — bóng bán kính r_i.
#       * Với mỗi nhóm điểm chưa gán trong B chia theo current_parent:
#           → Tạo node nội tại mới ở tầng i.
#           → Gán các điểm đó vào node mới; cập nhật current_parent.
#   - Điểm nào vẫn chưa được gán sau khi duyệt hết σ → tự thành singleton.
#
# Bước 4 — Gắn lá:
#   - Mỗi điểm p nhận một node lá riêng; cha của lá = current_parent[p]
#     (node nội tại cuối cùng được gán cho p tại tầng 0).
#
# Tính chất 2-HST được đảm bảo bởi cấu trúc phân cấp: node ở tầng i
# chỉ có con ở tầng i-1 (hoặc lá), và trọng số cạnh = 2^i.
# ===========================================================================


def _ttswd_tree_from_metric(
    D: np.ndarray, random_state=None, alpha=None,
) -> TTSWDTree:
    """Xây cây 2-HST từ ma trận metric đầy đủ D (thuật toán FRT).

    Dùng khi N ≤ 24 444 (chế độ "precompute"): toàn bộ ma trận D đã có sẵn
    trong RAM, truy vấn bóng O(n) bằng np.where(D[center] <= rad).

    Tham số
    -------
    D            : float64[n,n] — ma trận khoảng cách đối xứng, D[i,i]=0
    random_state : int | None   — seed cho RNG
    alpha        : float | None — nếu đặt: dùng beta=alpha thay vì Uniform[1,2]
    """
    rng = np.random.default_rng(random_state)
    n = D.shape[0]

    # --- Trường hợp biên: không có điểm ---
    if n == 0:
        return TTSWDTree(
            np.array([-1], np.int32),
            np.array([0], np.int16),
            np.zeros(0, np.int32),
            L=0,
        )

    diam = float(np.max(D)) if n else 0.0
    # L >= 1: đảm bảo vòng lặp tầng chạy ít nhất một lần (i = L-1 ≥ 0)
    L = max(1, int(np.ceil(np.log2(diam)))) if diam > 0 else 0

    # --- Trường hợp suy biến: tất cả điểm trùng nhau (diam = 0) ---
    # Tạo cây phẳng: gốc → n lá, tất cả cùng tầng 0.
    if diam <= 0:
        parent_buf = np.empty(1 + n, np.int32)
        level_buf = np.empty(1 + n, np.int16)
        parent_buf[0] = -1; level_buf[0] = L      # node 0 = gốc
        parent_buf[1:] = 0; level_buf[1:] = -1    # node 1..n = lá
        leaf_of_point = np.arange(1, 1 + n, dtype=np.int32)
        return TTSWDTree(parent_buf, level_buf, leaf_of_point, L=L)

    # --- Bước 1: Tham số hoá ngẫu nhiên ---
    # beta ~ Uniform[1,2]: dịch chuyển ranh giới phân vùng ngẫu nhiên
    beta = rng.uniform(1.0, 2.0) if alpha is None else float(alpha)
    # sigma: hoán vị ngẫu nhiên xác định thứ tự ưu tiên tâm phân cụm
    order = rng.permutation(n)

    # --- Bước 2: Cấp phát bộ nhớ ---
    # Số node tối đa: 1 gốc + tối đa n node nội tại × L tầng + n lá
    max_nodes = 1 + n * (L + 1)
    parent_buf = np.empty(max_nodes, np.int32)
    level_buf = np.empty(max_nodes, np.int16)
    parent_buf[0] = -1
    level_buf[0] = L      # node 0 = gốc ở tầng L
    nc = 1                # con trỏ node tiếp theo được cấp phát

    # current_parent[p]: node nội tại cha hiện tại của điểm p
    # Khởi tạo = 0 (gốc) vì ban đầu mọi điểm đều thuộc cây con của gốc
    current_parent = np.zeros(n, np.int32)
    leaf_of_point = np.empty(n, np.int32)
    # assigned[p]: id node nội tại đã được gán cho điểm p trong tầng hiện tại;
    # -1 = chưa được gán
    assigned = np.empty(n, np.int32)

    # --- Bước 3: Xây từng tầng ---
    # Tối ưu: khi mọi điểm đã là singleton ở tầng i, các tầng i-1, ..., 0
    # cũng sẽ là singleton → vector hoá thay vì duyệt bóng.
    all_singletons = False

    for i in range(L - 1, -1, -1):
        # Bán kính tầng i: r_i = β · 2^{i-1}  (< 2^i — thoả điều kiện 2-HST)
        rad = beta * (2.0 ** (i - 1))

        if all_singletons:
            # Tối ưu singleton: mỗi điểm nhận một node riêng — O(n) mảng
            node_ids = np.arange(nc, nc + n, dtype=np.int32)
            parent_buf[nc:nc + n] = current_parent  # cha = current_parent[p]
            level_buf[nc:nc + n] = i
            current_parent[:] = node_ids            # cập nhật cha cho tầng sau
            nc += n
            continue

        assigned.fill(-1)
        nc_before = nc
        n_assigned = 0

        # Duyệt tâm theo thứ tự σ: σ(0), σ(1), ...
        for l_idx in range(n):
            center = order[l_idx]          # σ(l)
            if assigned[center] != -1:     # σ(l) đã được gán → bỏ qua
                continue
            par_center = current_parent[center]
            in_ball = np.where(D[center] <= rad)[0]
            # Đúng theo bài báo: chỉ nhận điểm cùng cây con cha với center.
            # Tâm thuộc S₁ không được nhận điểm từ S₂ (cha khác).
            same_par = in_ball[current_parent[in_ball] == par_center]
            unassigned_in_ball = same_par[assigned[same_par] == -1]
            # Luôn chứa ít nhất center (size ≥ 1);
            # size == 0 chỉ xảy ra do sai số float32 — coi center là singleton.

            if unassigned_in_ball.size <= 1:
                # Singleton: chỉ có mình center trong cây con này
                parent_buf[nc] = par_center
                level_buf[nc] = i
                node_here = nc; nc += 1
                assigned[center] = node_here
                current_parent[center] = node_here
                n_assigned += 1
            else:
                # Tất cả điểm đã cùng par_center (do bộ lọc same_par)
                # → luôn tạo đúng 1 node, không cần xử lý đa cha
                parent_buf[nc] = par_center; level_buf[nc] = i
                node_here = nc; nc += 1
                assigned[unassigned_in_ball] = node_here
                current_parent[unassigned_in_ball] = node_here
                n_assigned += unassigned_in_ball.size

            if n_assigned >= n:
                break  # mọi điểm đã được gán — không cần duyệt thêm

        # Nếu tất cả điểm đều là singleton trong tầng này
        # → mọi tầng thấp hơn cũng sẽ là singleton (bán kính nhỏ hơn)
        if nc - nc_before == n:
            all_singletons = True

    # --- Bước 4: Gắn lá (vector hoá) ---
    # Mỗi điểm p nhận một node lá; cha = node nội tại cuối của p
    parent_buf[nc:nc + n] = current_parent
    level_buf[nc:nc + n] = -1                       # lá ở "tầng -1"
    leaf_of_point[:] = np.arange(nc, nc + n, dtype=np.int32)
    nc += n

    return TTSWDTree(
        parent_buf[:nc], level_buf[:nc], leaf_of_point,
        L=L,
    )


def _ttswd_tree_from_metric_streaming(
    coords_w: np.ndarray,
    random_state=None,
    alpha=None,
    block_size: int = 1024,
    diameter: Optional[float] = None,
    spatial_index: Optional[_SpatialIndex] = None,
) -> TTSWDTree:
    """Xây cây 2-HST dùng chỉ mục không gian cho truy vấn bóng (thuật toán FRT).

    Chế độ "streaming": thay vì giữ toàn bộ ma trận D (O(n²)) trong RAM,
    dùng cKDTree/BallTree để truy vấn bóng B = {j : d(center, j) ≤ r_i}
    với độ phức tạp O(log n + k) mỗi truy vấn, k = số điểm trong bóng.

    Truy vấn với eps=0 (mặc định) là chính xác: kết quả giống hệt
    np.where(D[center] <= rad) — đảm bảo lý thuyết FRT không bị xấp xỉ.

    Tham số
    -------
    coords_w      : float32[n,d]      — toạ độ đã nhân trọng số √wdim
    random_state  : int | None        — seed
    alpha         : float | None      — nếu đặt: dùng beta=alpha cố định
    block_size    : int               — kích thước khối khi tính đường kính
    diameter      : float | None      — đường kính đã tính trước (chia sẻ qua cây)
    spatial_index : _SpatialIndex | None — chỉ mục đã xây sẵn (chia sẻ qua cây)
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(coords_w, np.float32)
    n = X.shape[0]

    if n == 0:
        return TTSWDTree(
            np.array([-1], np.int32),
            np.array([0], np.int16),
            np.zeros(0, np.int32),
            L=0,
        )

    # Dùng đường kính chia sẻ nếu có — tránh tính lại O(n²/B) cho mỗi cây
    if diameter is not None:
        diam = float(diameter)
    else:
        diam = _streaming_diameter(X, block_size=block_size)

    L = max(1, int(np.ceil(np.log2(diam)))) if diam > 0 else 0

    if diam <= 0:
        parent_buf = np.empty(1 + n, np.int32)
        level_buf = np.empty(1 + n, np.int16)
        parent_buf[0] = -1; level_buf[0] = L
        parent_buf[1:] = 0; level_buf[1:] = -1
        leaf_of_point = np.arange(1, 1 + n, dtype=np.int32)
        return TTSWDTree(parent_buf, level_buf, leaf_of_point, L=L)

    # --- Bước 1: Tham số hoá ngẫu nhiên (giống chế độ precompute) ---
    beta = rng.uniform(1.0, 2.0) if alpha is None else float(alpha)
    order = rng.permutation(n)

    max_nodes = 1 + n * (L + 1)
    parent_buf = np.empty(max_nodes, np.int32)
    level_buf = np.empty(max_nodes, np.int16)
    parent_buf[0] = -1
    level_buf[0] = L
    nc = 1

    current_parent = np.zeros(n, np.int32)
    leaf_of_point = np.empty(n, np.int32)
    assigned = np.empty(n, np.int32)

    # Dùng chỉ mục không gian chia sẻ nếu có — xây một lần O(n log n),
    # dùng chung an toàn vì cKDTree/BallTree chỉ đọc sau khi xây.
    kd = spatial_index if spatial_index is not None else _SpatialIndex(X)

    # --- Bước 3: Xây từng tầng (logic giống _ttswd_tree_from_metric) ---
    all_singletons = False

    for i in range(L - 1, -1, -1):
        rad = beta * (2.0 ** (i - 1))

        if all_singletons:
            node_ids = np.arange(nc, nc + n, dtype=np.int32)
            parent_buf[nc:nc + n] = current_parent
            level_buf[nc:nc + n] = i
            current_parent[:] = node_ids
            nc += n
            continue

        assigned.fill(-1)
        nc_before = nc
        n_assigned = 0

        for l_idx in range(n):
            center = order[l_idx]
            if assigned[center] != -1:     # σ(l) đã được gán → bỏ qua
                continue
            par_center = current_parent[center]

            # Truy vấn bóng chính xác qua KDTree/BallTree — O(log n + k)
            # Thay thế np.where(D[center] <= rad) nhưng không cần ma trận D đầy đủ
            ball_list = kd.query_ball(X[center], float(rad))
            in_ball = np.array(ball_list, dtype=np.int32)

            # Lọc cùng cây con cha trước, sau đó lọc chưa gán — đúng theo bài báo
            same_par = in_ball[current_parent[in_ball] == par_center]
            unassigned_in_ball = same_par[assigned[same_par] == -1]
            # Luôn chứa ít nhất center (size ≥ 1)

            if unassigned_in_ball.size <= 1:
                # Singleton: chỉ có mình center trong cây con này
                parent_buf[nc] = par_center
                level_buf[nc] = i
                node_here = nc; nc += 1
                assigned[center] = node_here
                current_parent[center] = node_here
                n_assigned += 1
            else:
                # Tất cả điểm đã cùng par_center (do bộ lọc same_par)
                # → luôn tạo đúng 1 node
                parent_buf[nc] = par_center; level_buf[nc] = i
                node_here = nc; nc += 1
                assigned[unassigned_in_ball] = node_here
                current_parent[unassigned_in_ball] = node_here
                n_assigned += unassigned_in_ball.size

            if n_assigned >= n:
                break

        if nc - nc_before == n:
            all_singletons = True

    # --- Bước 4: Gắn lá ---
    parent_buf[nc:nc + n] = current_parent
    level_buf[nc:nc + n] = -1
    leaf_of_point[:] = np.arange(nc, nc + n, dtype=np.int32)
    nc += n

    return TTSWDTree(
        parent_buf[:nc], level_buf[:nc], leaf_of_point,
        L=L,
    )


def _ttswd_tree_unified(
    coords_w: np.ndarray,
    distance_mode: str = "auto",
    random_state: int | None = None,
    alpha: Optional[float] = None,
    block_size: int = 1024,
    precomputed_D: Optional[np.ndarray] = None,
    precomputed_diameter: Optional[float] = None,
    shared_spatial_index: Optional[_SpatialIndex] = None,
) -> Tuple[TTSWDTree, dict]:
    """Giao diện thống nhất: tự chọn chế độ precompute hoặc streaming.

    Chọn chế độ theo N và distance_mode:
      - "precompute" (N ≤ 24 444): dùng ma trận D đầy đủ, truy vấn O(n)
      - "streaming"  (N > 24 444): dùng KDTree/BallTree, truy vấn O(log n + k)

    Nhận các tài nguyên chia sẻ (precomputed_D, precomputed_diameter,
    shared_spatial_index) đã được tính một lần trong build_global_ttswd_forest
    và dùng chung cho tất cả cây trong forest.
    """
    t_start = time.perf_counter()
    selected_mode = _auto_select_mode(coords_w.shape[0], distance_mode, precompute_threshold=24444)
    meta = {"mode_used": selected_mode, "build_time": 0.0}

    if selected_mode == "precompute":
        D = precomputed_D if precomputed_D is not None else _pairwise_weighted_euclidean(coords_w, coords_w)
        tree = _ttswd_tree_from_metric(D, random_state=random_state, alpha=alpha)
    elif selected_mode == "streaming":
        tree = _ttswd_tree_from_metric_streaming(
            coords_w,
            random_state=random_state,
            alpha=alpha,
            block_size=block_size,
            diameter=precomputed_diameter,
            spatial_index=shared_spatial_index,
        )
    else:
        raise ValueError(f"Chế độ không hợp lệ: {selected_mode}")

    meta["build_time"] = time.perf_counter() - t_start
    return tree, meta
