import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="FreshTrack AI",
    page_icon="🥬",
    layout="wide"
)

# --------------------------------------------------
# 2. DASHBOARD HEADER
# --------------------------------------------------
st.title("🥬 FreshTrack AI")
st.subheader("Expiry Risk Monitoring and Food Waste Reduction")

st.info(
    "This prototype uses historical sales data and simulated "
    "inventory and expiry information."
)

# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------
DATA_FILE = BASE_DIR / "FreshTrack_AI_Expiry_Risk (1).csv"

if not DATA_FILE.exists():
    st.error(
        f"Dataset not found: {DATA_FILE.name}. "
        "Check the CSV filename in your GitHub repository."
    )
    st.stop()

df = pd.read_csv(DATA_FILE)

# Check required columns
required_columns = [
    "item_id",
    "avg_daily_sales",
    "stock_quantity",
    "days_to_expiry",
    "expected_unsold_stock",
    "expiry_risk"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# --------------------------------------------------
# 4. SUMMARY METRICS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    high_risk_count = (df["expiry_risk"] == "High").sum()
    st.metric("High-Risk Records", f"{high_risk_count:,}")

with col3:
    unsold_units = pd.to_numeric(
        df["expected_unsold_stock"], errors="coerce"
    ).fillna(0).sum()

    st.metric("Estimated Unsold Units", f"{unsold_units:,.0f}")

st.divider()

# --------------------------------------------------
# 5. EXPIRY RISK DISTRIBUTION
# --------------------------------------------------
st.subheader("Expiry Risk Distribution")

risk_counts = (
    df["expiry_risk"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
)

st.bar_chart(risk_counts)

# --------------------------------------------------
# 6. INVENTORY SEARCH AND FILTER
# --------------------------------------------------
st.subheader("Search Inventory")

search = st.text_input("Enter a product ID")

filtered = df.copy()

if search:
    filtered = filtered[
        filtered["item_id"].astype(str).str.contains(
            search, case=False, na=False
        )
    ]

risk_filter = st.selectbox(
    "Filter by expiry risk",
    ["All", "High", "Medium", "Low"]
)

if risk_filter != "All":
    filtered = filtered[
        filtered["expiry_risk"] == risk_filter
    ]

st.caption(f"Displaying {len(filtered):,} records")

st.dataframe(filtered, use_container_width=True)

# --------------------------------------------------
# 7. DOWNLOAD FILTERED RECORDS
# --------------------------------------------------
st.download_button(
    label="Download displayed records",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="FreshTrack_filtered_inventory.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 8. MACHINE-LEARNING EXPIRY RISK PREDICTION
# --------------------------------------------------
st.divider()
st.header("🤖 ML-Based Expiry Risk Prediction")

MODEL_FILE = BASE_DIR / "freshtrack_expiry_model.pkl"

if not MODEL_FILE.exists():
    st.warning(
        "The trained model file is missing. Upload "
        "freshtrack_expiry_model.pkl to your GitHub repository "
        "to enable predictions."
    )

else:
    try:
        model = joblib.load(MODEL_FILE)

        st.write(
            "Enter inventory details to estimate the expiry-risk "
            "category using the trained Random Forest model."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_daily_sales = st.number_input(
                "Average daily sales",
                min_value=0.0,
                value=5.0,
                step=0.5
            )

        with col2:
            stock_quantity = st.number_input(
                "Current stock quantity",
                min_value=0,
                value=50,
                step=1
            )

        with col3:
            days_to_expiry = st.number_input(
                "Days until expiry",
                min_value=0,
                value=10,
                step=1
            )

        if st.button("Predict Expiry Risk"):
            input_data = pd.DataFrame([{
                "avg_daily_sales": avg_daily_sales,
                "stock_quantity": stock_quantity,
                "days_to_expiry": days_to_expiry
            }])

            prediction = model.predict(input_data)[0]

            st.subheader(f"Predicted Expiry Risk: {prediction}")

            # Show probabilities when supported by the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]

                probability_df = pd.DataFrame({
                    "Risk category": model.classes_,
                    "Estimated probability (%)": (
                        probabilities * 100
                    ).round(2)
                })

                st.write("Model prediction probabilities")
                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True
                )

            st.warning(
                "Prototype only: the model was trained on simulated "
                "expiry-risk labels. Predictions are not validated "
                "for real-world supermarket inventory decisions."
            )

    except Exception as e:
        st.error(
            "The model could not be loaded or used. Check that the "
            "uploaded model was trained using the same three features."
        )
        st.caption(f"Technical details: {e}")
