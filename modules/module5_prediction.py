"""
Module 5: Influencer Prediction

This module:
1. Uses trained model
2. Predicts influencer status
3. Saves final output

Input:
    Model + dataframe (from Module 4)

Output:
    outputs/*_predictions.csv
"""

import os


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(model, df, output_path):
    print("\n🔹 Generating Predictions...")

    if model is None or df is None:
        print("❌ Model or data is missing!")
        return None

    # -------------------------------
    # SELECT FEATURES
    # -------------------------------
    X = df[["Degree", "PageRank", "Betweenness"]]

    # -------------------------------
    # MAKE PREDICTIONS
    # -------------------------------
    predictions = model.predict(X)

    df["Prediction"] = predictions

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"✅ Saved predictions to: {output_path}")

    # -------------------------------
    # DISPLAY SAMPLE
    # -------------------------------
    print("\nSample Output:\n", df.head())

    # -------------------------------
    # SUMMARY
    # -------------------------------
    total = len(df)
    influencers = int(df["Prediction"].sum())
    non_influencers = total - influencers

    print("\n📊 Summary:")
    print("Total Users:", total)
    print("Influencers:", influencers)
    print("Non-Influencers:", non_influencers)

    return df


# -------------------------------
# RUN MODULE INDEPENDENTLY
# -------------------------------
if __name__ == "__main__":

    from module4_model_training import train_model

    # Train model using sample features
    model, df = train_model("outputs/sample_features.csv")

    # Predict
    predict(model, df, "outputs/sample_predictions.csv")