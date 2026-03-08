"""Các metric đánh giá: MAP, độ chính xác phân cụm, chất lượng embedding."""
from __future__ import annotations

import numpy as np
from sklearn import neighbors
from sklearn.metrics import average_precision_score


def compute_map_knn_precomputed(X_computed, X_test_computed,
                                y_train, y_test, k=1):
    """Tính MAP cho k-NN (metric='precomputed') theo định nghĩa bài báo.

    - Khớp k-NN với k láng giềng.
    - predict_proba trên test → điểm số theo từng lớp.
    - Với mỗi lớp c: AP_c = average_precision_score(1_{y_test==c}, score_c)
    - MAP = trung bình AP_c qua tất cả lớp.

    Trả về MAP dưới dạng phần trăm [%].
    """
    clf = neighbors.KNeighborsClassifier(
        n_neighbors=k,
        metric="precomputed",
        weights="uniform",
    )
    clf.fit(X_computed, y_train)

    proba = clf.predict_proba(X_test_computed)   # shape (n_test, n_classes)
    classes = clf.classes_

    aps = []
    for c_idx, c in enumerate(classes):
        y_true_c = (y_test == c).astype(int)
        if np.sum(y_true_c) == 0:
            continue

        y_score_c = proba[:, c_idx]

        if np.all(y_score_c == y_score_c[0]):
            aps.append(0.0)
        else:
            aps.append(average_precision_score(y_true_c, y_score_c))

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps) * 100.0)


def _clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Độ chính xác phân cụm qua gán Hungarian.

    acc = max_hoán_vị sum_k |cluster_k ∩ class_σ(k)| / N
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    from sklearn.preprocessing import LabelEncoder
    y_true = LabelEncoder().fit_transform(np.asarray(y_true))
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    n_rows, n_cols = cm.shape
    if n_rows < n_cols:
        cm = np.vstack([cm, np.zeros((n_cols - n_rows, n_cols), dtype=cm.dtype)])
    elif n_cols < n_rows:
        cm = np.hstack([cm, np.zeros((n_rows, n_rows - n_cols), dtype=cm.dtype)])
    row_ind, col_ind = linear_sum_assignment(-cm)
    return float(cm[row_ind, col_ind].sum()) / float(len(y_true))


def _embedding_metrics(D_high: np.ndarray, embedding: np.ndarray,
                       n_neighbors: int = 5):
    """Tính trustworthiness, continuity và LCMC cho embedding 2D.

    Trustworthiness T(K): phạt các điểm gần trong embedding nhưng xa trong không gian gốc.
    Continuity     C(K): phạt các điểm gần trong không gian gốc nhưng xa trong embedding.
    LCMC           L(K): tỉ lệ láng giềng chung (Chen & Buja 2009).

    Cả ba đều trong [0, 1]; LCMC < 0 nghĩa là tệ hơn ngẫu nhiên.
    """
    from sklearn.metrics import pairwise_distances

    N = D_high.shape[0]
    K = min(n_neighbors, N - 1)
    if K == 0:
        return 1.0, 1.0, 0.0

    D_low = pairwise_distances(embedding)

    sort_high = np.argsort(D_high, axis=1)
    sort_low  = np.argsort(D_low,  axis=1)

    nn_high = sort_high[:, 1:K+1]
    nn_low  = sort_low[:,  1:K+1]

    rank_high = np.empty((N, N), dtype=np.int64)
    rank_high[np.arange(N)[:, None], sort_high] = np.arange(N)

    rank_low = np.empty((N, N), dtype=np.int64)
    rank_low[np.arange(N)[:, None], sort_low] = np.arange(N)

    trust_sum = 0.0
    cont_sum  = 0.0
    lcmc_sum  = 0

    for i in range(N):
        nh = set(nn_high[i])
        nl = set(nn_low[i])
        for j in nn_high[i]:
            if j not in nl:
                trust_sum += rank_low[i, j] - K
        for j in nn_low[i]:
            if j not in nh:
                cont_sum += rank_high[i, j] - K
        lcmc_sum += len(nh & nl)

    denom = N * K * (2 * N - 3 * K - 1)
    if denom > 0:
        norm  = 2.0 / denom
        trust = float(1.0 - norm * trust_sum)
        cont  = float(1.0 - norm * cont_sum)
    else:
        trust = cont = 1.0

    q_nx = lcmc_sum / (N * K)
    if N - 1 - K > 0:
        lcmc = float((N - 1) / (N - 1 - K) * (q_nx - K / (N - 1)))
    else:
        lcmc = float(q_nx)

    return trust, cont, lcmc
