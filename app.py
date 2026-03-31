import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Influencer Detection Dashboard",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data(name):
    return pd.read_csv(f"outputs/{name}_predictions.csv")


datasets = ["twitter", "retweet"]

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Influencer Detection Dashboard")
st.markdown("Graph-based analysis of social network influencers")

# -------------------------------
# DATASET SELECTOR
# -------------------------------
selected = st.selectbox("Select Dataset", datasets)

df = load_data(selected)

# -------------------------------
# METRICS
# -------------------------------
total = len(df)
influencers = int(df["Prediction"].sum())
non_influencers = total - influencers

col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Users", total)
col2.metric("⭐ Influencers", influencers)
col3.metric("👤 Non-Influencers", non_influencers)

st.markdown("---")

# -------------------------------
# CHARTS
# -------------------------------
st.subheader(f"📈 Analysis: {selected}")

counts = df["Prediction"].value_counts()

col1, col2 = st.columns(2)

# Bar chart
with col1:
    fig, ax = plt.subplots()
    ax.bar(
        ["Non-Influencers", "Influencers"],
        [counts.get(0, 0), counts.get(1, 0)]
    )
    ax.set_title("Distribution")
    st.pyplot(fig)

# Pie chart
with col2:
    fig, ax = plt.subplots()
    ax.pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=["Non", "Influencers"],
        autopct='%1.1f%%'
    )
    ax.set_title("Share")
    st.pyplot(fig)

# -------------------------------
# TOP INFLUENCERS
# -------------------------------
st.subheader("🔥 Top Influencers (PageRank)")

top_users = df.nlargest(10, "PageRank")
st.dataframe(top_users, use_container_width=True)

# -------------------------------
# COMPARISON SECTION
# -------------------------------
st.markdown("---")
st.header("📊 Dataset Comparison")

summary = []

for name in datasets:
    data = load_data(name)

    total = len(data)
    inf = int(data["Prediction"].sum())

    summary.append({
        "Dataset": name,
        "Total Users": total,
        "Influencers": inf,
        "Influencer %": round((inf / total) * 100, 2)
    })

summary_df = pd.DataFrame(summary)

st.dataframe(summary_df, use_container_width=True)

# -------------------------------
# COMPARISON BAR CHART
# -------------------------------
st.subheader("📈 Influencer % Comparison")

fig, ax = plt.subplots()
ax.bar(summary_df["Dataset"], summary_df["Influencer %"])
ax.set_ylabel("Percentage")
ax.set_title("Comparison Across Datasets")

st.pyplot(fig)

# -------------------------------
# INSIGHT BOX
# -------------------------------
avg = summary_df["Influencer %"].mean()

st.info(
    f"📌 Insight: On average, {avg:.2f}% of users are influencers. "
    "This shows that influence is concentrated among a small group of nodes."
)

# -------------------------------
# DOWNLOAD OPTION
# -------------------------------
st.subheader("⬇️ Download Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Selected Dataset",
    csv,
    f"{selected}_predictions.csv",
    "text/csv"
)