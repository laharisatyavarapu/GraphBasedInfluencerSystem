"""
Module 3: Feature Extraction

This module:
1. Takes graph as input
2. Computes centrality measures
3. Creates feature dataset

Features:
    - Degree Centrality
    - PageRank
    - Betweenness Centrality

Input:
    Graph (NetworkX)

Output:
    outputs/*_features.csv
"""

import pandas as pd
import networkx as nx
import os


# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------
def extract_features(G, output_path):
    print("\n🔹 Extracting Graph Features...")

    if G is None:
        print("❌ Graph is None!")
        return None

    # -------------------------------
    # DEGREE CENTRALITY
    # -------------------------------
    print("Calculating Degree Centrality...")
    degree = nx.degree_centrality(G)

    # -------------------------------
    # PAGERANK (MOST IMPORTANT)
    # -------------------------------
    print("Calculating PageRank...")
    pagerank = nx.pagerank(G)

    # -------------------------------
    # BETWEENNESS (APPROXIMATE)
    # -------------------------------
    print("Calculating Betweenness Centrality (approx)...")
    betweenness = nx.betweenness_centrality(G, k=50)

    # -------------------------------
    # CREATE DATAFRAME
    # -------------------------------
    df = pd.DataFrame({
        "User": list(G.nodes()),
        "Degree": list(degree.values()),
        "PageRank": list(pagerank.values()),
        "Betweenness": list(betweenness.values())
    })

    print("\nFeature Dataset Shape:", df.shape)

    # -------------------------------
    # SAVE FEATURES
    # -------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"✅ Saved features to: {output_path}")
    print("\nSample:\n", df.head())

    return df


# -------------------------------
# RUN MODULE INDEPENDENTLY
# -------------------------------
if __name__ == "__main__":

    from module2_graph_building import build_graph

    # Load sample graph
    G = build_graph("outputs/sample_clean.csv")

    # Extract features
    extract_features(G, "outputs/sample_features.csv")