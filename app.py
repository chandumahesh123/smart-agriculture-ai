import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Smart Agriculture Advisory", layout="wide")

st.title("🌾 Smart Agriculture Yield & Advisory System")

# -------------------------
# DATASET GENERATION
# -------------------------

crops = [
    "Rice","Wheat","Maize","Bajra","Jowar","Ragi","Barley",
    "Chickpea","PigeonPea","Lentil","Mungbean","BlackGram",
    "Soybean","Groundnut","Mustard","Sunflower","Safflower","Sesame","Castor",
    "Sugarcane","Cotton","Jute","Tobacco",
    "Tea","Coffee","Rubber","Coconut","Cashew",
    "Mango","Banana","Potato","Onion","Tomato","Spices"
]

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
        base_yield
        + 0.015*K
        + 0.0004*Ca
        + 0.001*Mg
        - 0.4*Na
        + 0.008*P
        + 0.004*S
        + 0.00005*Fe
        + 0.015*Zn
    ) / 10

    yield_value = max(1, min(yield_value, 12))

    data.append([crop, soil, K, Ca, Mg, Na, P, S, Fe, Zn, Mn, B, yield_value])

columns = ["Crop","Soil","K","Ca","Mg","Na","P","S","Fe","Zn","Mn","B","Yield"]
df = pd.DataFrame(data, columns=columns)

# Encoding
crop_encoder = LabelEncoder()
soil_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Soil"] = soil_encoder.fit_transform(df["Soil"])

X = df.drop("Yield", axis=1)
y = df["Yield"]

model = RandomForestRegressor()
model.fit(X, y)

# -------------------------
# UI SECTION
# -------------------------

st.header("Enter Soil & Nutrient Values")

col1, col2 = st.columns(2)

with col1:
    crop_input = st.selectbox("Select Crop", crop_encoder.classes_)
    soil_input = st.selectbox("Select Soil Type", soil_encoder.classes_)
    K = st.number_input("Potassium (ppm)", 0.0)
    Ca = st.number_input("Calcium (ppm)", 0.0)
    Mg = st.number_input("Magnesium (ppm)", 0.0)
    Na = st.number_input("Sodium (%)", 0.0)

with col2:
    P = st.number_input("Phosphorus (ppm)", 0.0)
    S = st.number_input("Sulfur (ppm)", 0.0)
    Fe = st.number_input("Iron (ppm)", 0.0)
    Zn = st.number_input("Zinc (ppm)", 0.0)
    Mn = st.number_input("Manganese (ppm)", 0.0)
    B = st.number_input("Boron (ppm)", 0.0)

if st.button("Predict Yield"):

    temp = {
        "Crop": crop_encoder.transform([crop_input])[0],
        "Soil": soil_encoder.transform([soil_input])[0],
        "K": K, "Ca": Ca, "Mg": Mg, "Na": Na,
        "P": P, "S": S, "Fe": Fe, "Zn": Zn,
        "Mn": Mn, "B": B
    }

    input_array = np.array(list(temp.values())).reshape(1,-1)
    predicted_yield = model.predict(input_array)[0]

    st.success(f"Estimated Yield: {round(predicted_yield,2)} tons/hectare")

    fig, ax = plt.subplots()
    ax.bar(["Estimated Yield"], [predicted_yield])
    ax.set_ylabel("Yield (tons/hectare)")
    st.pyplot(fig)
