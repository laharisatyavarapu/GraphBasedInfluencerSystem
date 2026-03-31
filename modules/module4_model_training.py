"""
Module 4: Model Training

This module:
1. Loads feature dataset
2. Creates labels based on PageRank
3. Trains a machine learning model
4. Evaluates performance

Input:
    outputs/*_features.csv

Output:
    Trained model + labeled dataframe
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------------------
# TRAIN MODEL FUNCTION
# -------------------------------
def train_model(feature_path):
    print(f"\n🔹 Training Model: {feature_path}")

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    try:
        df = pd.read_csv(feature_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None

    print("Dataset Shape:", df.shape)

    # -------------------------------
    # CREATE LABELS (TOP 10%)
    # -------------------------------
    threshold = df["PageRank"].quantile(0.90)

    df["Label"] = (df["PageRank"] >= threshold).astype(int)

    print("✅ Labels created (Top 10% influencers)")

    # -------------------------------
    # FEATURES & TARGET
    # -------------------------------
    X = df[["Degree", "PageRank", "Betweenness"]]
    y = df["Label"]

    # -------------------------------
    # TRAIN-TEST SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✅ Model trained successfully!")

    # -------------------------------
    # EVALUATE MODEL
    # -------------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"📊 Model Accuracy: {accuracy:.4f}")

    print("\nSample Data:\n", df.head())

    return model, df


# -------------------------------
# RUN MODULE INDEPENDENTLY
# -------------------------------
if __name__ == "__main__":

    train_model("outputs/sample_features.csv")