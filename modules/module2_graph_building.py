"""
Module 2: Graph Building

This module:
1. Loads cleaned dataset
2. Builds a directed graph using NetworkX
3. Displays graph statistics

Input:
    outputs/*_clean.csv

Output:
    NetworkX Graph object
"""

import pandas as pd
import networkx as nx


# -------------------------------
# BUILD GRAPH FUNCTION
# -------------------------------
def build_graph(input_path):
    print(f"\n🔹 Building Graph: {input_path}")

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

    print("Dataset Shape:", df.shape)

    # -------------------------------
    # CREATE GRAPH
    # -------------------------------
    try:
        G = nx.from_pandas_edgelist(
            df,
            source="source",
            target="target",
            create_using=nx.DiGraph()
        )
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        return None

    # -------------------------------
    # GRAPH STATISTICS
    # -------------------------------
    print("\n📊 Graph Statistics:")
    print("Number of Nodes:", G.number_of_nodes())
    print("Number of Edges:", G.number_of_edges())

    return G


# -------------------------------
# RUN MODULE INDEPENDENTLY
# -------------------------------
if __name__ == "__main__":

    G = build_graph("outputs/sample_clean.csv")

    if G:
        print("\n✅ Graph created successfully!")