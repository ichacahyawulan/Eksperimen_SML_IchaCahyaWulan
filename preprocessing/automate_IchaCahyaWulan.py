# automate_IchaCahyawulan.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[INFO] Duplikat dihapus: {before - after} baris. Shape sekarang: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values dengan menghapus baris yang memilikinya."""
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[INFO] Missing values dihapus: {before - after} baris. Shape sekarang: {df.shape}")
    return df


def normalize_features(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Normalisasi fitur numerik menggunakan StandardScaler."""
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"[INFO] Normalisasi selesai untuk kolom: {numerical_cols}")
    return df


def split_and_save(df: pd.DataFrame, target_col: str, output_path: str,
                   test_size: float = 0.2, random_state: int = 42) -> str:
    """Melakukan train-test split dan menyimpan hasil preprocessing."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_data = X_train.copy()
    train_data[target_col] = y_train.values
    test_data = X_test.copy()
    test_data[target_col] = y_test.values

    df_final = pd.concat([train_data, test_data], ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"[INFO] Dataset preprocessing disimpan di: {output_path}")
    print(f"[INFO] Shape final: {df_final.shape}")
    return output_path


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Fungsi utama pipeline preprocessing.
    Mengembalikan DataFrame yang sudah siap dilatih.
    """
    NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    TARGET_COL = 'target'

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = normalize_features(df, NUMERICAL_COLS)
    split_and_save(df, TARGET_COL, output_path)

    print("[INFO] Preprocessing selesai!")
    return df


if __name__ == "__main__":
    INPUT_PATH = "../heart_raw.csv"
    OUTPUT_PATH = "./heart_preprocessing.csv"
    preprocess(INPUT_PATH, OUTPUT_PATH)