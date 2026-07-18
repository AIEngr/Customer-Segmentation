"""
Customer Segmentation Explorer — Streamlit App
==========================================================
Loads a trained KMeans model (kmeans.pkl, k=3) and assigns a new customer
to one of three behavioral segments based on Age, Average_Spend,
Visits_per_Week, and Promotion_Interest.

NOTE ON LABEL MAPPING:
The source notebook defines two different mappings from cluster id to
segment name:
  1) `cluster_names = {0: "Daily", 1: "Promotion", 2: "Weekend"}`, which is
     what was actually used to create the `Customer Group` column.
  2) The `clustering()` helper function, which used different if/elif
     logic (0 -> Daily, 1 -> Weekend, else -> Promotion).
These two disagree on clusters 1 and 2. This app uses mapping #1, since
that's the one that produced the labeled training data. Verify against
your own cluster analysis before trusting this mapping in production.

Expects `kmeans.pkl` in the same directory as this file.

Run:
    streamlit run app.py

Python: 3.12
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Explorer",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "kmeans.pkl"

# Authoritative cluster label mapping — see module docstring above.
CLUSTER_NAMES = {0: "Daily", 1: "Promotion", 2: "Weekend"}

CLUSTER_INFO = {
    "Daily": {
        "emoji": "📅",
        "color": "#3498db",
        "desc": "Frequent, steady shoppers who visit often but aren't strongly promotion-driven.",
    },
    "Promotion": {
        "emoji": "🏷️",
        "color": "#e67e22",
        "desc": "Deal-driven customers with high interest in promotions and offers.",
    },
    "Weekend": {
        "emoji": "🌤️",
        "color": "#9b59b6",
        "desc": "Occasional shoppers who visit less frequently, often around weekends.",
    },
}

FEATURES = ["Age", "Average_Spend", "Visits_per_Week", "Promotion_Interest"]


# --------------------------------------------------------------------------
# Model / data loading (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_demo_data():
    """Recreate the synthetic training set (fixed seed = 42) purely for
    visualization context — this is demo data, not persisted production data."""
    np.random.seed(42)
    data = {
        "CustomerID": np.arange(1, 101),
        "Age": np.random.randint(18, 65, size=100),
        "Average_Spend": np.random.uniform(5, 50, size=100),
        "Visits_per_Week": np.random.uniform(1, 7, size=100),
        "Promotion_Interest": np.random.randint(1, 11, size=100),
    }
    return pd.DataFrame(data)


model = load_model()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🛍️ Customer Segmentation Explorer")
st.markdown(
    "Assign a customer to one of three behavioral segments using a "
    "**K-Means (k=3)** clustering model."
)
st.divider()

if model is None:
    st.error(
        f"Model file not found at `{MODEL_PATH.name}`. "
        "Place `kmeans.pkl` in the same folder as this app, then restart Streamlit."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app uses **K-Means (k=3)** trained on four customer
        attributes to identify shopping behavior segments:

        - 📅 **Daily** — frequent, steady shoppers
        - 🏷️ **Promotion** — deal-driven shoppers
        - 🌤️ **Weekend** — occasional shoppers
        """
    )
    st.divider()
    st.caption("Built with scikit-learn + Streamlit · Python 3.12")

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
st.subheader("Enter Customer Profile")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=35)
    avg_spend = st.slider(
        "Average Spend ($)", min_value=5.0, max_value=50.0, value=25.0, step=0.5
    )
with col2:
    visits = st.slider(
        "Visits per Week", min_value=1.0, max_value=7.0, value=3.5, step=0.1
    )
    promo_interest = st.slider("Promotion Interest (1–10)", min_value=1, max_value=10, value=5)

st.divider()
predict_clicked = st.button(
    "Assign Segment", type="primary", width="stretch"
)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if predict_clicked:
    # Build as a DataFrame with the same column names KMeans was fitted on
    # (df[['Age', 'Average_Spend', 'Visits_per_Week', 'Promotion_Interest']]
    # in the training notebook). Predicting on a bare numpy array triggers
    # sklearn's "X does not have valid feature names" warning.
    new_customer_df = pd.DataFrame([[age, avg_spend, visits, promo_interest]], columns=FEATURES)
    new_customer = new_customer_df.to_numpy()
    cluster_id = int(model.predict(new_customer_df)[0])
    group_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    info = CLUSTER_INFO.get(group_name, {"emoji": "ℹ️", "color": "#7f8c8d", "desc": ""})

    st.markdown(
        f"""
        <div style="padding: 1.2rem; border-radius: 0.6rem;
                    background-color: {info['color']}22;
                    border: 2px solid {info['color']}; text-align: center;">
            <span style="font-size: 1.4rem;">{info['emoji']} Assigned Segment:
            <strong style="color: {info['color']};">{group_name}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"_{info['desc']}_")

    # Distance to each cluster center as a rough confidence signal
    st.markdown("#### Distance to Cluster Centers")
    centers = model.cluster_centers_
    distances = np.linalg.norm(centers - new_customer, axis=1)
    dist_df = pd.DataFrame(
        {
            "Segment": [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(len(centers))],
            "Distance": distances,
        }
    ).sort_values("Distance")
    st.dataframe(
        dist_df.style.format({"Distance": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption("Lower distance = stronger match to that segment's typical profile.")

    # Visualize the new customer against demo training data
    st.markdown("#### Where This Customer Falls (Age vs. Average Spend)")
    demo_df = load_demo_data()
    demo_df["Cluster"] = model.predict(demo_df[FEATURES])
    demo_df["Segment"] = demo_df["Cluster"].map(CLUSTER_NAMES)

    fig, ax = plt.subplots(figsize=(7, 5))
    for seg, group in demo_df.groupby("Segment"):
        color = CLUSTER_INFO.get(seg, {}).get("color", "#7f8c8d")
        ax.scatter(
            group["Age"], group["Average_Spend"], label=seg, alpha=0.6, color=color, s=60
        )

    ax.scatter(
        age,
        avg_spend,
        marker="*",
        s=400,
        color="black",
        edgecolor="white",
        linewidth=1.5,
        label="This Customer",
        zorder=5,
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Spend ($)")
    ax.set_title("Customer Segments")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "Reference points are the original synthetic training data (fixed random "
        "seed), shown for context only. In production, persist a real snapshot "
        "of scored customers instead of regenerating synthetic data."
    )

    with st.expander("View Submitted Profile"):
        st.json(
            {
                "Age": age,
                "Average_Spend": avg_spend,
                "Visits_per_Week": visits,
                "Promotion_Interest": promo_interest,
            }
        )