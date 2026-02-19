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

from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Smart Agriculture System", layout="wide")
st.title("🌾 Smart Agriculture Yield & Leaf Disease Advisory System")

# =====================================================
# CACHE LEAF MODEL
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
# GENERATE DATASET FOR YIELD
# =====================================================

data = []

for _ in range(2000):
    K = random.randint(80, 300)
    Ca = random.randint(500, 40000)
    Mg = random.randint(1000, 16000)
    P = random.randint(50, 2500)
    S = random.randint(50, 2500)
    Zn = random.randint(5, 300)

    yield_value = (
        4 + 0.015*K + 0.0004*Ca + 0.001*Mg +
        0.008*P + 0.004*S + 0.015*Zn
    ) / 10

    yield_value = max(1, min(yield_value, 12))
    data.append([K, Ca, Mg, P, S, Zn, yield_value])

df = pd.DataFrame(data, columns=["K","Ca","Mg","P","S","Zn","Yield"])

X = df.drop("Yield", axis=1)
y = df["Yield"]

yield_model = RandomForestRegressor()
yield_model.fit(X,y)

# =====================================================
# USER INPUT
# =====================================================

st.header("🧪 Enter Soil Nutrient Values")

K = st.number_input("Potassium (ppm)", 0.0)
Ca = st.number_input("Calcium (ppm)", 0.0)
Mg = st.number_input("Magnesium (ppm)", 0.0)
P = st.number_input("Phosphorus (ppm)", 0.0)
S = st.number_input("Sulfur (ppm)", 0.0)
Zn = st.number_input("Zinc (ppm)", 0.0)

st.subheader("🌿 Leaf Disease Detection")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg","webp"])

# =====================================================
# BUTTON
# =====================================================

if st.button("🔍 Predict Yield & Advisory"):

    # ---------------- Yield Prediction ----------------
    input_df = pd.DataFrame([[K,Ca,Mg,P,S,Zn]],
                            columns=["K","Ca","Mg","P","S","Zn"])

    before_yield = yield_model.predict(input_df)[0]

    # Simple correction logic
    corrected_df = input_df.copy()

    means = X.mean()

    for col in corrected_df.columns:
        if corrected_df[col][0] < means[col]:
            corrected_df[col] = means[col]

    after_yield = yield_model.predict(corrected_df)[0]

    improvement = ((after_yield - before_yield) / max(before_yield,0.01)) * 100

    st.subheader("📋 Yield Prediction Report")
    st.write("Yield Before Correction:", round(before_yield,2), "tons/hectare")
    st.write("Yield After Correction:", round(after_yield,2), "tons/hectare")
    st.write("Expected Improvement:", round(improvement,2), "%")

    # Yield Graph
    fig, ax = plt.subplots()
    ax.bar(["Before","After"], [before_yield, after_yield], color=["orange","green"])
    ax.set_ylabel("Yield (tons/hectare)")
    ax.set_title("Yield Improvement Analysis")
    st.pyplot(fig)

    # ---------------- Leaf Detection ----------------
    if uploaded_file is not None:

        st.image(uploaded_file, caption="Uploaded Leaf Image")

        disease, conf = predict_leaf_disease(uploaded_file)

        st.subheader("🌿 Leaf Disease Analysis Report")
        st.write("Detected Condition:", disease)
        st.write("Confidence:", round(conf,2), "%")

        if conf < 80:
            severity = "Mild"
        elif conf < 90:
            severity = "Moderate"
        else:
            severity = "Severe"

        st.write("Severity Level:", severity)

        if disease != "Healthy":
            st.error("⚠ DISEASE DETECTED")
            st.write("Recommended Fungicide: Carbendazim")
        else:
            st.success("Leaf is Healthy")

    else:
        st.warning("Upload leaf image for detection.")
