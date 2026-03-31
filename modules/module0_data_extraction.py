"""
Module 0: Data Extraction & Conversion

This module:
1. Extracts twitter.zip (ego network dataset)
2. Merges all .edges files into one dataset
3. Converts rt-retweet.mtx into edge list
4. Saves all datasets in a common CSV format (source, target)

Output:
    data/processed/twitter.csv
    data/processed/retweet.csv
"""

import os
import zipfile
import pandas as pd
from scipy.io import mmread


# -------------------------------
# CREATE REQUIRED FOLDERS
# -------------------------------
def create_folders():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


# -------------------------------
# EXTRACT ZIP FILE
# -------------------------------
def extract_zip(zip_path, extract_path):
    print(f"\n🔄 Extracting ZIP: {zip_path}")

    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        return False

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"✅ Extracted to: {extract_path}")
    return True


# -------------------------------
# MERGE ALL .edges FILES
# -------------------------------
def process_edges(folder_path, output_csv):
    print(f"\n🔄 Processing .edges files from: {folder_path}")

    all_edges = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".edges"):
                file_path = os.path.join(root, file)

                try:
                    df = pd.read_csv(file_path, sep=" ", names=["source", "target"])
                    all_edges.append(df)
                    print(f"   Loaded: {file}")
                except Exception as e:
                    print(f"   ⚠️ Skipped {file}: {e}")

    if len(all_edges) == 0:
        print("❌ No .edges files found!")
        return

    combined_df = pd.concat(all_edges, ignore_index=True)

    # Clean data
    combined_df = combined_df.drop_duplicates()
    combined_df = combined_df[combined_df["source"] != combined_df["target"]]

    combined_df.to_csv(output_csv, index=False)

    print(f"✅ Saved merged dataset: {output_csv}")
    print("Shape:", combined_df.shape)


# -------------------------------
# FIND MTX FILE
# -------------------------------
def find_mtx(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mtx"):
                return os.path.join(root, file)
    return None


# -------------------------------
# MTX → CSV
# -------------------------------
def process_mtx(mtx_path, output_csv):
    print(f"\n🔄 Converting MTX → CSV: {mtx_path}")

    matrix = mmread(mtx_path).tocoo()

    df = pd.DataFrame({
        "source": matrix.row,
        "target": matrix.col
    })

    df = df.drop_duplicates()
    df = df[df["source"] != df["target"]]

    df.to_csv(output_csv, index=False)

    print(f"✅ Saved MTX dataset: {output_csv}")
    print("Shape:", df.shape)


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def run_data_extraction():
    print("🚀 Starting Data Extraction (FINAL)...\n")

    create_folders()

    # -------- TWITTER ZIP (EDGES) --------
    twitter_zip = "data/raw/twitter.zip"
    twitter_extract = "data/temp/twitter"
    twitter_output = "data/processed/twitter.csv"

    if extract_zip(twitter_zip, twitter_extract):
        process_edges(twitter_extract, twitter_output)

    # -------- RETWEET ZIP (MTX) --------
    retweet_zip = "data/raw/rt-retweet.zip"
    retweet_extract = "data/temp/retweet"
    retweet_output = "data/processed/retweet.csv"

    if extract_zip(retweet_zip, retweet_extract):
        mtx_file = find_mtx(retweet_extract)

        if mtx_file:
            process_mtx(mtx_file, retweet_output)
        else:
            print("❌ No .mtx file found!")

    print("\n🎉 DATA EXTRACTION COMPLETED!")


# -------------------------------
# RUN MODULE
# -------------------------------
if __name__ == "__main__":
    run_data_extraction()