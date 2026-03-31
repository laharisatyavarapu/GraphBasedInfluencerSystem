"""
Module 1: Data Preprocessing

This module:
1. Loads processed CSV data
2. Cleans the dataset
3. Removes duplicates and self-loops
4. Ensures correct column format
5. Saves cleaned dataset

Input:
    data/processed/*.csv

Output:
    outputs/*_clean.csv
"""

import pandas as pd
import os


# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess_data(input_path, output_path, sample_size=None):
    print(f"\n🔹 Preprocessing: {input_path}")

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print("Original Shape:", df.shape)

    # -------------------------------
    # STANDARDIZE COLUMN NAMES
    # -------------------------------
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ["source", "target"]
    else:
        print("❌ Invalid data format!")
        return

    # -------------------------------
    # REMOVE MISSING VALUES
    # -------------------------------
    df = df.dropna()

    # -------------------------------
    # REMOVE SELF LOOPS
    # -------------------------------
    df = df[df["source"] != df["target"]]

    # -------------------------------
    # REMOVE DUPLICATES
    # -------------------------------
    df = df.drop_duplicates()

    # -------------------------------
    # OPTIONAL SAMPLING (PERFORMANCE)
    # -------------------------------
    if sample_size:
        df = df.head(sample_size)
        print(f"⚡ Sampled to {sample_size} rows")

    print("Cleaned Shape:", df.shape)

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"✅ Saved cleaned data to: {output_path}")
    print("\nSample:\n", df.head())

    return df


# -------------------------------
# RUN MODULE INDEPENDENTLY
# -------------------------------
if __name__ == "__main__":

    # Example test run
    preprocess_data(
        input_path="data/processed/twitter.csv",
        output_path="outputs/sample_clean.csv",
        sample_size=10000   # optional
    )