"""Tiện ích tải dữ liệu và lưu/đọc bảng kết quả."""
from __future__ import annotations

import os

import pandas as pd


def _has_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401
        return True
    except Exception:
        return False


def _load_results_table(out_file, cols):
    """Đọc bảng kết quả từ file Excel hoặc CSV; trả về (DataFrame, đường_dẫn, dùng_excel)."""
    excel_ok = _has_openpyxl()
    csv_file = os.path.splitext(out_file)[0] + ".csv"

    if excel_ok and os.path.exists(out_file):
        try:
            return pd.read_excel(out_file, engine="openpyxl"), out_file, True
        except Exception:
            return pd.read_excel(out_file), out_file, True

    if os.path.exists(csv_file):
        return pd.read_csv(csv_file), (out_file if excel_ok else csv_file), excel_ok

    if os.path.exists(out_file) and not excel_ok:
        print(f"[WARN] openpyxl chưa cài; không đọc được {out_file}. Ghi vào {csv_file}.")

    return pd.DataFrame(columns=cols), (out_file if excel_ok else csv_file), excel_ok


def _save_results_table(df, out_file, use_excel):
    """Lưu bảng kết quả ra file Excel hoặc CSV."""
    if use_excel:
        df.to_excel(out_file, index=False, engine="openpyxl")
    else:
        df.to_csv(out_file, index=False)


def load_ucr_dataset_tsl(data_dir, dataset_name):
    """Tải dữ liệu train/test từ file .ts cục bộ (không tải từ mạng).

    Tham số
    ----------
    data_dir     : thư mục gốc chứa các thư mục dataset
    dataset_name : tên dataset UCR

    Kết quả
    -------
    X_train, y_train, X_test, y_test
    """
    from tslearn.datasets import UCR_UEA_datasets

    abs_data_dir = os.path.abspath(data_dir)
    train_path = os.path.join(abs_data_dir, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_path  = os.path.join(abs_data_dir, dataset_name, f"{dataset_name}_TEST.ts")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' không có local tại {abs_data_dir}. "
            f"Hãy tải thủ công và đặt vào {os.path.join(abs_data_dir, dataset_name)}/"
        )

    ucr = UCR_UEA_datasets()
    ucr._data_dir = abs_data_dir
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)

    print("Đã tải dataset:", dataset_name)
    print("Kích thước train:", len(y_train))
    print("Kích thước test:", len(y_test))

    return X_train, y_train, X_test, y_test
