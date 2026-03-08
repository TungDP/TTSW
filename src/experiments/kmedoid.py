"""Thực nghiệm phân cụm k-Medoids."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from ttswd import stack_union_points_with_time, build_global_ttswd_forest, compute_distance_matrices_ttswd

from .io import load_ucr_dataset_tsl, _load_results_table, _save_results_table
from .metrics import _clustering_accuracy


def run_kmedoid(datapath, datatype, alg,
                n_clusters=None,
                num_runs=5,
                result_dir="../result",
                alg_kwargs=None):
    """Chạy k-Medoids với khoảng cách tính trước qua nhiều lần chạy.

    Mỗi lần chạy:
      - Tính ma trận khoảng cách train–train.
      - Chạy k-Medoids trên train với k = n_clusters (mặc định = số lớp).
      - Đánh giá bằng NMI và độ chính xác phân cụm.
    Sau num_runs lần: báo cáo trung bình và phương sai của nmi, acc, thời gian chạy.

    Kết quả ghi vào <result_dir>/TTSWD_kmedoid.xlsx (hoặc .csv).
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:
        raise ImportError(
            "run_kmedoid yêu cầu scikit-learn-extra. "
            "Cài đặt bằng: pip install scikit-learn-extra"
        ) from exc

    _kw = alg_kwargs or {}

    if isinstance(datapath, list):
        results = {}
        for ds in datapath:
            try:
                results[ds] = run_kmedoid(ds, datatype, alg,
                                          n_clusters=n_clusters,
                                          num_runs=num_runs,
                                          result_dir=result_dir,
                                          alg_kwargs=alg_kwargs)
            except FileNotFoundError as e:
                print(f"[BỎ QUA] {ds}: {e}")
        return results

    if datatype == "UCR_TSL":
        X_train, y_train, X_test, y_test = load_ucr_dataset_tsl("../data/UCR", datapath)
    else:
        raise ValueError(f"Loại dữ liệu không hỗ trợ: {datatype}")

    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    train_len = len(y_all)
    k = n_clusters if n_clusters is not None else int(len(np.unique(y_all)))

    nmi_list  = []
    acc_list  = []
    time_list = []

    for run_idx in range(num_runs):
        print(f"\n========== Lần {run_idx + 1}/{num_runs} — {alg} k-Medoids trên {datapath} ==========")
        t0 = time.time()

        if alg == "TTSWD":
            X_train_series = [np.asarray(X_all[i], dtype=float) for i in range(train_len)]
            coords_all, _, seq_boundaries = stack_union_points_with_time(
                X_train_series, [], normalize_time=True
            )
            trees, _ = build_global_ttswd_forest(
                coords_all,
                n_trees=_kw.get("n_trees", 32),
                time_weight=_kw.get("time_weight", 64.0),
                distance_scale=_kw.get("distance_scale", 1.0),
                random_state=run_idx * 100,
                distance_mode=_kw.get("distance_mode", "auto"),
                streaming_block_size=_kw.get("streaming_block_size", 2048),
            )
            X_computed, _ = compute_distance_matrices_ttswd(
                trees, seq_boundaries, train_len, 0
            )
        else:
            raise ValueError(f"Thuật toán không hỗ trợ: {alg}. Hỗ trợ: TTSWD")

        maxv = float(np.max(X_computed)) if X_computed.size else 0.0
        if maxv > 0:
            X_computed /= maxv

        kmed = KMedoids(n_clusters=k, metric="precomputed",
                        random_state=run_idx, max_iter=300)
        kmed.fit(X_computed)
        labels = kmed.labels_

        nmi     = float(normalized_mutual_info_score(y_all, labels, average_method="arithmetic"))
        acc     = _clustering_accuracy(y_all, labels)
        runtime = time.time() - t0

        print(f"[Lần {run_idx+1}] NMI={nmi:.4f}  ACC={acc:.4f}  t={runtime:.2f}s")

        nmi_list.append(nmi)
        acc_list.append(acc)
        time_list.append(runtime)

    nmi_mean  = float(np.mean(nmi_list))
    nmi_var   = float(np.var(nmi_list))
    acc_mean  = float(np.mean(acc_list))
    acc_var   = float(np.var(acc_list))
    time_mean = float(np.mean(time_list))
    time_var  = float(np.var(time_list))

    print("\n========== Tổng kết ==========")
    print(f"NMI     : mean={nmi_mean:.4f}, var={nmi_var:.6f}")
    print(f"ACC     : mean={acc_mean:.4f}, var={acc_var:.6f}")
    print(f"Thời gian: mean={time_mean:.2f} s, var={time_var:.4f}")

    dataset_key = f"{datapath}_{datatype}"
    os.makedirs(result_dir, exist_ok=True)
    out_file = os.path.join(result_dir, f"{alg}_kmedoid.xlsx")
    cols = ["dataset", "nmi_mean", "nmi_var", "acc_mean", "acc_var",
            "runtime_mean", "runtime_var"]
    new_row = {
        "dataset":      dataset_key,
        "nmi_mean":     nmi_mean,  "nmi_var":     nmi_var,
        "acc_mean":     acc_mean,  "acc_var":     acc_var,
        "runtime_mean": time_mean, "runtime_var": time_var,
    }

    df, out_file_actual, use_excel = _load_results_table(out_file, cols)
    if "dataset" not in df.columns:
        df.insert(0, "dataset", "")

    mask = df["dataset"] == dataset_key
    if mask.any():
        for c in cols:
            df.loc[mask, c] = new_row[c]
    else:
        df = pd.concat(
            [df, pd.DataFrame([new_row], columns=cols)],
            ignore_index=True,
        )

    df = df[cols]
    _save_results_table(df, out_file_actual, use_excel)

    return {
        "nmi_mean":     nmi_mean,  "nmi_var":     nmi_var,
        "acc_mean":     acc_mean,  "acc_var":     acc_var,
        "runtime_mean": time_mean, "runtime_var": time_var,
        "nmi_runs":     nmi_list,
        "acc_runs":     acc_list,
        "time_runs":    time_list,
    }
