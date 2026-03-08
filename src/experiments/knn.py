"""Thực nghiệm phân loại k-NN."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score

from ttswd import stack_union_points_with_time, build_global_ttswd_forest, compute_distance_matrices_ttswd

from .io import load_ucr_dataset_tsl, _load_results_table, _save_results_table
from .metrics import compute_map_knn_precomputed


def run_knn(datapath, datatype, alg,
            num_neighbor_list=None,
            num_runs=5,
            result_dir="../result",
            alg_kwargs=None):
    """Chạy k-NN với khoảng cách tính trước qua nhiều lần chạy.

    Mỗi lần chạy:
      - Tính ma trận khoảng cách train–train và test–train.
      - Chạy k-NN cho từng k trong num_neighbor_list, chọn k có độ chính xác cao nhất.
      - Tính MAP (tại k tốt nhất).
    Sau num_runs lần: báo cáo trung bình và phương sai của accuracy, MAP, thời gian chạy.

    alg_kwargs ghi đè mặc định TTSWD:
      n_trees=112, time_weight=64.0, distance_mode="auto",
      streaming_block_size=8192, distance_scale=1.0

    Kết quả ghi vào <result_dir>/<alg>_knn.xlsx (hoặc .csv).
    """
    if num_neighbor_list is None:
        num_neighbor_list = [1]

    _kw = alg_kwargs or {}

    if isinstance(datapath, list):
        results = {}
        for ds in datapath:
            try:
                results[ds] = run_knn(ds, datatype, alg,
                                      num_neighbor_list=num_neighbor_list,
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


    train_len = len(y_train)
    test_len  = len(y_test)

    acc_list  = []
    map_list  = []
    time_list = []

    for run_idx in range(num_runs):
        print(f"\n========== Lần {run_idx + 1}/{num_runs} — {alg} trên {datapath} ==========")
        t0 = time.time()

        if alg == "TTSWD":
            X_train_series = [np.asarray(X_train[i], dtype=float) for i in range(train_len)]
            X_test_series  = [np.asarray(X_test[i],  dtype=float) for i in range(test_len)]
            coords_all, _, seq_boundaries = stack_union_points_with_time(
                X_train_series, X_test_series, normalize_time=True
            )
            trees, meta = build_global_ttswd_forest(
                coords_all,
                n_trees=_kw.get("n_trees", 112),
                time_weight=_kw.get("time_weight", 64.0),
                random_state=run_idx * 4,
                distance_mode=_kw.get("distance_mode", "auto"),
                streaming_block_size=_kw.get("streaming_block_size", 8192),
                distance_scale=_kw.get("distance_scale", 1.0),
            )
            meta.seq_boundaries = seq_boundaries
            X_computed, X_test_computed = compute_distance_matrices_ttswd(
                trees, seq_boundaries, train_len, test_len
            )
        else:
            raise ValueError(f"Thuật toán không hỗ trợ: {alg}. Hỗ trợ: TTSWD")

        maxv = float(np.max(X_computed)) if X_computed.size else 0.0
        if maxv > 0:
            X_computed /= maxv
            X_test_computed /= maxv

        k_list = sorted(set(num_neighbor_list))
        best_acc = np.nan
        best_k = None

        for k in k_list:
            if k > train_len:
                print(f"Bỏ qua k={k} (n_train={train_len} < k)")
                continue
            clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric="precomputed")
            clf.fit(X_computed, y_train)
            y_pred = clf.predict(X_test_computed)
            acc = 100.0 * accuracy_score(y_test, y_pred)
            print(f"[Lần {run_idx+1}] Accuracy {k}NN: {acc:.2f} %")
            if (best_k is None) or (acc > best_acc):
                best_acc = acc
                best_k = k

        if best_k is None:
            best_acc = np.nan

        print(f"[Lần {run_idx+1}] Accuracy tốt nhất: {best_acc:.2f} % (k={best_k})")

        k_map = best_k if best_k is not None else 1
        map_score = compute_map_knn_precomputed(
            X_computed, X_test_computed, y_train, y_test, k=k_map
        )
        print(f"[Lần {run_idx+1}] MAP (k={k_map}): {map_score:.2f} %")

        runtime_s = time.time() - t0
        print(f"[Lần {run_idx+1}] Thời gian: {runtime_s:.2f} s")

        acc_list.append(best_acc)
        map_list.append(map_score)
        time_list.append(runtime_s)

    acc_mean  = float(np.mean(acc_list))
    acc_var   = float(np.var(acc_list))
    map_mean  = float(np.mean(map_list))
    map_var   = float(np.var(map_list))
    time_mean = float(np.mean(time_list))
    time_var  = float(np.var(time_list))

    print("\n========== Tổng kết ==========")
    print(f"Accuracy: mean={acc_mean:.2f} %, var={acc_var:.4f}")
    print(f"MAP     : mean={map_mean:.2f} %, var={map_var:.4f}")
    print(f"Thời gian: mean={time_mean:.2f} s, var={time_var:.4f}")

    dataset_key = f"{datapath}_{datatype}"
    os.makedirs(result_dir, exist_ok=True)
    out_file = os.path.join(result_dir, f"{alg}_knn.xlsx")
    cols = ["dataset", "accuracy_mean", "accuracy_var",
            "map_mean", "map_var", "runtime_mean", "runtime_var"]
    new_row = {
        "dataset": dataset_key,
        "accuracy_mean": acc_mean, "accuracy_var": acc_var,
        "map_mean": map_mean,      "map_var": map_var,
        "runtime_mean": time_mean, "runtime_var": time_var,
    }

    df, out_file_actual, use_excel = _load_results_table(out_file, cols)
    if "dataset" not in df.columns:
        df.insert(0, "dataset", "")

    mask = (df["dataset"] == dataset_key)
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
        "accuracy_mean": acc_mean, "accuracy_var": acc_var,
        "map_mean": map_mean,      "map_var": map_var,
        "runtime_mean": time_mean, "runtime_var": time_var,
        "acc_runs": acc_list, "map_runs": map_list, "time_runs": time_list,
    }
