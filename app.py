# Machine Learning Prediction
st.divider()
st.header("🤖 ML-Based Expiry Risk Prediction")

try:
    model = joblib.load(
        Path(__file__).parent / "freshtrack_expiry_model.pkl"
    )

    st.write("Enter inventory details to predict expiry risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_daily_sales = st.number_input(
            "Average daily sales",
            min_value=0.0,
            value=5.0
        )

    with col2:
        stock_quantity = st.number_input(
            "Current stock quantity",
            min_value=0,
            value=50
        )

    with col3:
        days_to_expiry = st.number_input(
            "Days until expiry",
            min_value=0,
            value=10
        )

    if st.button("Predict Expiry Risk"):
        input_data = pd.DataFrame([{
            "avg_daily_sales": avg_daily_sales,
            "stock_quantity": stock_quantity,
            "days_to_expiry": days_to_expiry
        }])

        prediction = model.predict(input_data)[0]

        st.subheader(f"Predicted Expiry Risk: {prediction}")

        st.caption(
            "Prototype prediction based on simulated training labels; "
            "not validated for real-world inventory decisions."
        )

except FileNotFoundError:
    st.warning(
        "Upload freshtrack_expiry_model.pkl to the GitHub repository "
        "to enable ML predictions."
    )
