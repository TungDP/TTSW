"""Thực nghiệm trực quan hóa t-SNE."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from ttswd import stack_union_points_with_time, build_global_ttswd_forest, compute_distance_matrices_ttswd

from .io import load_ucr_dataset_tsl, _load_results_table, _save_results_table
from .metrics import _embedding_metrics


def run_tsne(datapath, datatype, alg,
             num_runs=5,
             n_components=2,
             result_dir="../result",
             alg_kwargs=None):
    """Chạy t-SNE với khoảng cách tính trước qua nhiều lần chạy.

    Mỗi lần chạy:
      - Tính ma trận khoảng cách train–train.
      - Chạy t-SNE (metric='precomputed') → embedding 2D.
      - Đánh giá: trustworthiness, continuity, LCMC.
    Lần chạy đầu (seed 0): lưu biểu đồ scatter vào <result_dir>/<dataset>/.
    Sau num_runs lần: báo cáo mean/var và ghi vào <alg>_tsne.xlsx.

    alg_kwargs mặc định (TTSWD):
      n_trees=32, time_weight=64.0, distance_mode="auto",
      streaming_block_size=2048, distance_scale=1.0
    """
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _kw = alg_kwargs or {}

    if isinstance(datapath, list):
        results = {}
        for ds in datapath:
            try:
                results[ds] = run_tsne(ds, datatype, alg,
                                       num_runs=num_runs,
                                       n_components=n_components,
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

    train_len  = len(y_all)
    perplexity = min(30, max(5, train_len - 1))

    trust_list = []
    cont_list  = []
    lcmc_list  = []
    time_list  = []

    for run_idx in range(num_runs):
        print(f"\n========== Lần {run_idx + 1}/{num_runs} — {alg} t-SNE trên {datapath} ==========")
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
                random_state=run_idx * 4,
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

        # Đảm bảo ma trận khoảng cách đối xứng hoàn toàn
        X_computed = (X_computed + X_computed.T) / 2.0
        np.fill_diagonal(X_computed, 0.0)

        seed = run_idx * 4
        tsne = TSNE(
            n_components=n_components,
            metric="precomputed",
            perplexity=perplexity,
            random_state=seed,
            init="random",
            n_iter=1000,
        )
        embedding = tsne.fit_transform(X_computed)

        if run_idx == 0:
            ds_result_dir = os.path.join(result_dir, datapath)
            os.makedirs(ds_result_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            classes = np.unique(y_all)
            cmap = plt.get_cmap("tab10", len(classes))
            for ci, cls in enumerate(classes):
                mask = y_all == cls
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=str(cls), s=15, alpha=0.7, color=cmap(ci)
                )
            ax.set_title(f"{alg} t-SNE — {datapath} (seed 0)", fontsize=12)
            ax.legend(markerscale=2, fontsize=8, loc="best")
            ax.set_xlabel("Chiều 1")
            ax.set_ylabel("Chiều 2")
            plt.tight_layout()
            plot_path = os.path.join(ds_result_dir, f"{alg}_tsne_seed0.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[BIỂU ĐỒ] Đã lưu → {plot_path}")

        trust, cont, lcmc = _embedding_metrics(
            X_computed, embedding,
            n_neighbors=min(5, train_len - 1),
        )

        runtime = time.time() - t0
        print(f"[Lần {run_idx+1}] Trust={trust:.4f}  Cont={cont:.4f}  LCMC={lcmc:.4f}  t={runtime:.2f}s")

        trust_list.append(trust)
        cont_list.append(cont)
        lcmc_list.append(lcmc)
        time_list.append(runtime)

    trust_mean = float(np.mean(trust_list))
    trust_var  = float(np.var(trust_list))
    cont_mean  = float(np.mean(cont_list))
    cont_var   = float(np.var(cont_list))
    lcmc_mean  = float(np.mean(lcmc_list))
    lcmc_var   = float(np.var(lcmc_list))
    time_mean  = float(np.mean(time_list))
    time_var   = float(np.var(time_list))

    print("\n========== Tổng kết ==========")
    print(f"Trustworth   : mean={trust_mean:.4f}, var={trust_var:.6f}")
    print(f"Continuity   : mean={cont_mean:.4f}, var={cont_var:.6f}")
    print(f"LCMC         : mean={lcmc_mean:.4f}, var={lcmc_var:.6f}")
    print(f"Thời gian    : mean={time_mean:.2f} s, var={time_var:.4f}")

    dataset_key = f"{datapath}_{datatype}"
    os.makedirs(result_dir, exist_ok=True)
    out_file = os.path.join(result_dir, f"{alg}_tsne.xlsx")
    cols = [
        "dataset",
        "trust_mean", "trust_var",
        "cont_mean", "cont_var",
        "lcmc_mean", "lcmc_var",
        "runtime_mean", "runtime_var",
    ]
    new_row = {
        "dataset":      dataset_key,
        "trust_mean":   trust_mean, "trust_var":   trust_var,
        "cont_mean":    cont_mean,  "cont_var":    cont_var,
        "lcmc_mean":    lcmc_mean,  "lcmc_var":    lcmc_var,
        "runtime_mean": time_mean,  "runtime_var": time_var,
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
        "trust_mean":   trust_mean, "trust_var":   trust_var,
        "cont_mean":    cont_mean,  "cont_var":    cont_var,
        "lcmc_mean":    lcmc_mean,  "lcmc_var":    lcmc_var,
        "runtime_mean": time_mean,  "runtime_var": time_var,
        "trust_runs":   trust_list,
        "cont_runs":    cont_list,
        "lcmc_runs":    lcmc_list,
        "time_runs":    time_list,
    }
