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

st.set_page_config(page_title="Smart Agriculture System", layout="wide")
st.title("🌾 Smart Agriculture Yield & Leaf Disease Advisory System")

# =====================================================
# LOAD LEAF MODEL (Cached – Prevent Reload Issue)
# =====================================================

@st.cache_resource
def load_leaf_model():

    leaf_classes = ["Healthy", "Leaf Blight", "Powdery Mildew", "Leaf Spot"]

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(leaf_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model, leaf_classes

leaf_model, leaf_classes = load_leaf_model()

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
# SIMPLE YIELD MODEL (Stable)
# =====================================================

df = pd.DataFrame({
    "K": np.random.randint(80,300,1000),
    "Ca": np.random.randint(500,40000,1000),
    "Mg": np.random.randint(1000,16000,1000),
    "Yield": np.random.uniform(2,6,1000)
})

X = df[["K","Ca","Mg"]]
y = df["Yield"]

model = RandomForestRegressor()
model.fit(X,y)

# =====================================================
# USER INPUT
# =====================================================

st.header("🧪 Enter Nutrient Values")

K = st.number_input("Potassium", 0.0)
Ca = st.number_input("Calcium", 0.0)
Mg = st.number_input("Magnesium", 0.0)

st.subheader("🌿 Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg","webp"])

# =====================================================
# BUTTON
# =====================================================

if st.button("🔍 Predict"):

    # Yield Prediction
    input_df = pd.DataFrame([[K,Ca,Mg]], columns=["K","Ca","Mg"])
    predicted_yield = model.predict(input_df)[0]

    st.subheader("📋 Yield Prediction")
    st.write("Predicted Yield:", round(predicted_yield,2), "tons/hectare")

    # Leaf Prediction
    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Leaf Image")

        disease, conf = predict_leaf_disease(uploaded_file)

        st.subheader("🌿 Leaf Disease Analysis Report")
        st.write("Detected Condition:", disease)
        st.write("Confidence:", round(conf,2), "%")

        if disease != "Healthy":
            st.error("⚠ DISEASE DETECTED")
            st.write("Recommended Fungicide: Carbendazim")
        else:
            st.success("Leaf is Healthy")

    else:
        st.warning("Upload leaf image for detection.")
