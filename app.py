import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Consistent Results
# ----------------------------
np.random.seed(42)
tf.random.set_seed(42)

st.set_page_config(page_title="Smart Agriculture System", layout="wide")
st.title("🌾 Smart Agriculture Yield & Leaf Disease Advisory System")

# =====================================================
# DATASET GENERATION
# =====================================================

crops = ["Rice","Wheat","Maize","Bajra","Jowar","Ragi","Barley",
         "Chickpea","PigeonPea","Lentil","Mungbean","BlackGram",
         "Soybean","Groundnut","Mustard","Sunflower","Safflower",
         "Sesame","Castor","Sugarcane","Cotton","Jute","Tobacco",
         "Tea","Coffee","Rubber","Coconut","Cashew",
         "Mango","Banana","Potato","Onion","Tomato","Spices"]

soils = ["Clay","Sandy","Loamy","Black","Red","Alluvial"]

data = []

for _ in range(3000):
    crop = random.choice(crops)
    soil = random.choice(soils)

    K = random.randint(80, 300)
    Ca = random.randint(500, 40000)
    Mg = random.randint(1000, 16000)
    Na = random.uniform(0, 8)
    P = random.randint(50, 2500)
    S = random.randint(50, 2500)
    Fe = random.randint(4000, 60000)
    Zn = random.randint(5, 300)
    Mn = random.randint(100, 12000)
    B = random.randint(2, 200)

    base_yield = 4
    if crop == "Sugarcane":
        base_yield = 6
    elif crop in ["Tea","Coffee","Rubber"]:
        base_yield = 5

    yield_value = (
        base_yield + 0.015*K + 0.0004*Ca + 0.001*Mg
        - 0.4*Na + 0.008*P + 0.004*S
        + 0.00005*Fe + 0.015*Zn
    ) / 10

    yield_value = max(1, min(yield_value, 12))
    data.append([crop, soil, K, Ca, Mg, Na, P, S, Fe, Zn, Mn, B, yield_value])

columns = ["Crop","Soil","K","Ca","Mg","Na","P","S","Fe","Zn","Mn","B","Yield"]
df = pd.DataFrame(data, columns=columns)

# =====================================================
# ENCODING
# =====================================================

crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil"] = soil_encoder.fit_transform(df["Soil"])

X = df.drop("Yield", axis=1)
y = df["Yield"]

# =====================================================
# MODEL TRAINING
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = {}
metrics_table = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_score = cross_val_score(model, X, y, cv=5).mean()

    results[name] = r2
    metrics_table.append([name, mae, rmse, r2, cv_score])

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

# =====================================================
# LEAF DISEASE MODEL (Same as Colab)
# =====================================================

leaf_classes = ["Healthy", "Leaf Blight", "Powdery Mildew", "Leaf Spot"]

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(leaf_classes), activation='softmax')(x)

leaf_model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

def predict_leaf_disease(uploaded_image):

    img = Image.open(uploaded_image).resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = leaf_model.predict(img_array)

    predicted_class = leaf_classes[np.argmax(preds)]
    confidence = np.max(preds) * 100

    return predicted_class, confidence

# =====================================================
# USER INPUT
# =====================================================

st.header("🧪 Enter Soil & Nutrient Values")

crop_input = st.selectbox("Select Crop", crop_encoder.classes_)
soil_input = st.selectbox("Select Soil Type", soil_encoder.classes_)
K = st.number_input("Potassium (ppm)", 0.0)
Ca = st.number_input("Calcium (ppm)", 0.0)
Mg = st.number_input("Magnesium (ppm)", 0.0)
Na = st.number_input("Sodium (%)", 0.0)
P = st.number_input("Phosphorus (ppm)", 0.0)
S = st.number_input("Sulfur (ppm)", 0.0)
Fe = st.number_input("Iron (ppm)", 0.0)
Zn = st.number_input("Zinc (ppm)", 0.0)
Mn = st.number_input("Manganese (ppm)", 0.0)
B = st.number_input("Boron (ppm)", 0.0)

st.subheader("🌿 Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg","webp"])

# =====================================================
# PREDICTION
# =====================================================

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

    st.subheader("📊 Model Performance")
    metrics_df = pd.DataFrame(
        metrics_table,
        columns=["Model","MAE","RMSE","R2 Score","Cross Val Score"]
    )
    st.dataframe(metrics_df)
    st.success(f"Best Model Selected: {best_model_name}")

    st.subheader("📋 Yield Prediction")
    st.write("Predicted Yield:", round(before_yield,2), "tons/hectare")

    # ================= Leaf Detection =================

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
