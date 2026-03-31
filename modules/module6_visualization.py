"""
Module 6: Dataset Comparison & Visualization

This module:
1. Loads prediction datasets
2. Compares influencer distribution
3. Generates visualizations

Input:
    outputs/twitter_predictions.csv
    outputs/retweet_predictions.csv

Output:
    Visual charts + summary
"""

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except:
        print(f"❌ Missing file: {path}")
        return None


# -------------------------------
# CREATE SUMMARY
# -------------------------------
def create_summary(datasets):
    summary = []

    for name, path in datasets.items():
        df = load_data(path)

        if df is None:
            continue

        total = len(df)
        influencers = int(df["Prediction"].sum())

        summary.append({
            "Dataset": name,
            "Total Users": total,
            "Influencers": influencers,
            "Influencer %": round((influencers / total) * 100, 2)
        })

    return pd.DataFrame(summary)


# -------------------------------
# BAR CHART
# -------------------------------
def plot_bar(summary_df):
    print("\n📊 Generating Bar Chart...")

    plt.figure()

    plt.bar(
        summary_df["Dataset"],
        summary_df["Influencer %"]
    )

    plt.xlabel("Dataset")
    plt.ylabel("Influencer %")
    plt.title("Influencer Comparison Across Datasets")

    plt.show()


# -------------------------------
# PIE CHART
# -------------------------------
def plot_pie(df, dataset_name):
    print(f"\n📊 Generating Pie Chart: {dataset_name}")

    counts = df["Prediction"].value_counts()

    plt.figure()

    plt.pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=["Non-Influencers", "Influencers"],
        autopct="%1.1f%%"
    )

    plt.title(f"{dataset_name} Distribution")

    plt.show()


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def compare_datasets():
    print("\n🚀 Comparing Datasets...\n")

    datasets = {
        "Twitter": "outputs/twitter_predictions.csv",
        "Retweet": "outputs/retweet_predictions.csv"
    }

    # Create summary
    summary_df = create_summary(datasets)

    if summary_df.empty:
        print("❌ No data available for comparison!")
        return

    print("📊 Summary:\n", summary_df)

    # Plot bar chart
    plot_bar(summary_df)

    # Plot pie charts
    for name, path in datasets.items():
        df = load_data(path)
        if df is not None:
            plot_pie(df, name)

    print("\n✅ Visualization Complete!")


# -------------------------------
# RUN MODULE
# -------------------------------
if __name__ == "__main__":
    compare_datasets()