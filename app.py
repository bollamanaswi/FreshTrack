
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="FreshTrack AI",
    page_icon="🥬",
    layout="wide"
)

st.title("🥬 FreshTrack AI")
st.subheader("Expiry Risk Monitoring and Food Waste Reduction")

st.info(
    "This prototype uses historical sales data and "
    "simulated inventory and expiry information."
)

# Load the exported dataset
df = pd.read_csv("/content/FreshTrack_AI_Expiry_Risk.csv")

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric(
        "High-Risk Records",
        f"{(df['expiry_risk'] == 'High').sum():,}"
    )

with col3:
    st.metric(
        "Estimated Unsold Units",
        f"{df['expected_unsold_stock'].sum():,.0f}"
    )

st.divider()

# Risk distribution
st.subheader("Expiry Risk Distribution")

risk_counts = (
    df["expiry_risk"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
)

st.bar_chart(risk_counts)

# Product search
st.subheader("Search Inventory")

search = st.text_input("Enter a product ID")

filtered = df.copy()

if search:
    filtered = filtered[
        filtered["item_id"].str.contains(search, case=False, na=False)
    ]

# Risk filter
risk_filter = st.selectbox(
    "Filter by expiry risk",
    ["All", "High", "Medium", "Low"]
)

if risk_filter != "All":
    filtered = filtered[filtered["expiry_risk"] == risk_filter]

st.dataframe(filtered, use_container_width=True)

# Download filtered results
st.download_button(
    "Download displayed records",
    data=filtered.to_csv(index=False),
    file_name="FreshTrack_filtered_inventory.csv",
    mime="text/csv"
)
