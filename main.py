"""
Main Pipeline: Graph-Based Influencer Detection

This script:
1. Runs preprocessing
2. Builds graph
3. Extracts features
4. Trains model
5. Generates predictions

Datasets:
    - twitter (from .edges)
    - retweet (from .mtx)

Outputs:
    outputs/twitter_predictions.csv
    outputs/retweet_predictions.csv
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try importing, else install
try:
    import pandas
except:
    install("pandas")

try:
    import networkx
except:
    install("networkx")

try:
    import sklearn
except:
    install("scikit-learn")

try:
    import matplotlib
except:
    install("matplotlib")

try:
    import streamlit
except:
    install("streamlit")

try:
    import scipy
except:
    install("scipy")

from modules.module1_preprocessing import preprocess_data
from modules.module2_graph_building import build_graph
from modules.module3_feature_extraction import extract_features
from modules.module4_model_training import train_model
from modules.module5_prediction import predict


# -------------------------------
# DATASET PATHS
# -------------------------------
datasets = {
    "twitter": "data/processed/twitter.csv",
    "retweet": "data/processed/retweet.csv"
}


# -------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------
def run_pipeline():
    print("🚀 Starting Full Pipeline...\n")

    for name, path in datasets.items():

        print("\n==============================")
        print(f"Processing Dataset: {name}")
        print("==============================")

        # Define output paths
        clean_path = f"outputs/{name}_clean.csv"
        feature_path = f"outputs/{name}_features.csv"
        prediction_path = f"outputs/{name}_predictions.csv"

        # -------------------------------
        # MODULE 1: PREPROCESSING
        # -------------------------------
        df_clean = preprocess_data(
            input_path=path,
            output_path=clean_path,
            sample_size=20000   # optional for performance
        )

        # -------------------------------
        # MODULE 2: GRAPH BUILDING
        # -------------------------------
        G = build_graph(clean_path)

        # -------------------------------
        # MODULE 3: FEATURE EXTRACTION
        # -------------------------------
        df_features = extract_features(G, feature_path)

        # -------------------------------
        # MODULE 4: MODEL TRAINING
        # -------------------------------
        model, df_model = train_model(feature_path)

        # -------------------------------
        # MODULE 5: PREDICTION
        # -------------------------------
        df_pred = predict(model, df_model, prediction_path)

        print(f"\n✅ Completed: {name}")

    print("\n🎉 ALL DATASETS PROCESSED SUCCESSFULLY!")


# -------------------------------
# RUN SCRIPT
# -------------------------------
if __name__ == "__main__":
    run_pipeline()