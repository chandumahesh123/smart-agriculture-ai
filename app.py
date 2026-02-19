if st.button("🔍 Predict Yield & Advisory"):

    encoded_crop = crop_encoder.transform([crop_input])[0]
    encoded_soil = soil_encoder.transform([soil_input])[0]

    input_data = {
        "Crop": encoded_crop,
        "Soil": encoded_soil,
        "K": K, "Ca": Ca, "Mg": Mg, "Na": Na,
        "P": P, "S": S, "Fe": Fe, "Zn": Zn,
        "Mn": Mn, "B": B
    }

    input_df = pd.DataFrame([input_data])
    before_yield = best_model.predict(input_df)[0]

    # ================= Model Performance =================

    st.subheader("📊 Model Performance")

    metrics_df = pd.DataFrame(
        metrics_table,
        columns=["Model","MAE","RMSE","R2 Score","Cross Val Score"]
    )

    st.dataframe(metrics_df)
    st.success(f"Best Model Selected: {best_model_name}")

    # ================= Advisory System =================

    optimal_means = X.mean()
    suggestions = []
    corrected_data = input_data.copy()

    for nutrient in ["K","Ca","Mg","P","S","Zn","B"]:
        if input_data[nutrient] < optimal_means[nutrient]:
            suggestions.append(
                f"{nutrient} LOW → Increase to approx {round(optimal_means[nutrient],1)}"
            )
            corrected_data[nutrient] = optimal_means[nutrient]

    corrected_df = pd.DataFrame([corrected_data])
    after_yield = best_model.predict(corrected_df)[0]

    improvement = ((after_yield - before_yield) / max(before_yield, 0.01)) * 100
    improvement = max(improvement, 0)

    # ================= Yield Report =================

    st.subheader("📋 Farmer Advisory Report")

    st.write("**Crop:**", crop_input)
    st.write("**Soil:**", soil_input)

    if suggestions:
        st.write("### Suggested Corrections:")
        for s in suggestions:
            st.write("-", s)
    else:
        st.success("All nutrients are within optimal range.")

    st.write("Yield Before Correction:", round(before_yield,2), "tons/hectare")
    st.write("Yield After Correction:", round(after_yield,2), "tons/hectare")
    st.write("Expected Improvement:", round(improvement,2), "%")

    # Yield Graph
    fig, ax = plt.subplots()
    ax.bar(["Before","After"], [before_yield, after_yield], color=["orange","green"])
    ax.set_ylabel("Yield (tons/hectare)")
    ax.set_title("Yield Improvement Analysis")
    st.pyplot(fig)

    # ================= Leaf Disease Detection =================

    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

        disease, conf = predict_leaf_disease(uploaded_file)

        st.subheader("🌿 Leaf Disease Analysis Report")

        st.write("Detected Condition :", disease)
        st.write("Prediction Confidence :", round(conf,2), "%")

        if conf < 80:
            severity = "Mild"
        elif conf < 90:
            severity = "Moderate"
        else:
            severity = "Severe"

        fungicide_recommendations = {
            "Leaf Blight": {
                "fungicide": "Mancozeb",
                "dosage": "2.5 grams per liter",
                "purpose": "Controls blight-causing fungi."
            },
            "Powdery Mildew": {
                "fungicide": "Carbendazim",
                "dosage": "1 gram per liter",
                "purpose": "Effective against powdery mildew fungus."
            },
            "Leaf Spot": {
                "fungicide": "Copper Oxychloride",
                "dosage": "3 grams per liter",
                "purpose": "Controls fungal leaf spot infections."
            }
        }

        if disease != "Healthy":

            st.error(f"⚠ DISEASE DETECTED: {disease}")
            st.write("Severity Level:", severity)

            info = fungicide_recommendations.get(disease)

            if info:
                st.write("### 🧪 Recommended Fungicide")
                st.write("Product :", info["fungicide"])
                st.write("Dosage  :", info["dosage"])
                st.write("Purpose :", info["purpose"])

            st.write("### 🌿 Additional Treatment Advice")
            st.write("- Remove infected leaves immediately")
            st.write("- Avoid overhead irrigation")
            st.write("- Ensure proper plant spacing")
            st.write("- Monitor crop weekly")

        else:
            st.success("Leaf is Healthy. No fungicide required.")

    else:
        st.warning("Please upload a leaf image.")
