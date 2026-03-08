"""Shim tương thích ngược — tái xuất từ gói experiments.

Tất cả tiện ích thí nghiệm đã được chuyển vào gói `experiments`:
  - experiments.io       : load_ucr_dataset_tsl
  - experiments.metrics  : compute_map_knn_precomputed, _clustering_accuracy, _embedding_metrics
  - experiments.knn      : run_knn
  - experiments.kmedoid  : run_kmedoid
  - experiments.tsne     : run_tsne

"""
import os
import sys

# Đảm bảo src/ có trong path khi chạy utilities.py trực tiếp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.io import load_ucr_dataset_tsl
from experiments.metrics import (
    compute_map_knn_precomputed,
    _clustering_accuracy,
    _embedding_metrics,
)
from experiments.knn import run_knn
from experiments.kmedoid import run_kmedoid
from experiments.tsne import run_tsne

__all__ = [
    "load_ucr_dataset_tsl",
    "compute_map_knn_precomputed",
    "_clustering_accuracy",
    "_embedding_metrics",
    "run_knn",
    "run_kmedoid",
    "run_tsne",
]
