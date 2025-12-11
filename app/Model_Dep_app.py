import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------------------------------------
# Load your trained model and scaler
# ---------------------------------------------
model = pickle.load(open("rand_forest.pkl", "rb"))   # replace file name
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="Liver Disease Classifier", layout="wide")
st.title("🧬 Liver Disease Classification App")

st.write("Upload patient data for batch prediction or adjust parameters manually for single prediction.")

# ---------------------------------------------
# Batch Prediction Section
# ---------------------------------------------
st.subheader("📂 Batch Prediction from Excel File")
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# Define feature names
feature_names = ['albumin', 'alkaline_phosphatase', 'alanine_aminotransferase',
                 'aspartate_aminotransferase', 'bilirubin', 'cholinesterase',
                 'cholesterol', 'gamma_glutamyl_transferase', 'protein']

# Class map
class_map = {0: 'cirrhosis', 1: 'fibrosis', 2: 'hepatitis', 3: 'no_disease', 4: 'suspect_disease'}

if uploaded_file is not None:
    batch_df = pd.read_excel(uploaded_file)

    # Ensure correct column order
    batch_inputs = batch_df[feature_names]

    # Scale inputs
    batch_scaled = pd.DataFrame(scaler.transform(batch_inputs), columns=feature_names)

    # Predict
    batch_predictions = model.predict(batch_scaled)
    batch_proba = model.predict_proba(batch_scaled)

    batch_df['Predicted_Class'] = [class_map[p] for p in batch_predictions]

    st.write("### 🩺 Batch Prediction Results")
    st.dataframe(batch_df)

    # Show probability distribution for first few rows
    st.write("### 📊 Probability Distribution (First 5 Patients)")
    prob_df = pd.DataFrame(batch_proba[:5], columns=[class_map[i] for i in range(len(class_map))])
    st.dataframe(prob_df)

# ---------------------------------------------
# Manual Input Section
# ---------------------------------------------
st.subheader("✍️ Manual Input for Single Prediction")
st.sidebar.header("🔧 Input Features")

albumin = st.sidebar.number_input('Albumin (g/L)', min_value=27.8, max_value=62.9, value=38.5, step=0.1)
alkaline_phosphatase = st.sidebar.number_input('Alkaline Phosphatase (IU/L)', min_value=27.0, max_value=145.0, value=52.5, step=0.1)
alanine_aminotransferase = st.sidebar.number_input('Alanine Aminotransferase (IU/L)', min_value=7.0, max_value=118.0, value=7.7, step=0.1)
aspartate_aminotransferase = st.sidebar.number_input('Aspartate Aminotransferase (IU/L)', min_value=12.0, max_value=69.0, value=22.1, step=0.1)
bilirubin = st.sidebar.number_input('Bilirubin (µmol/L)', min_value=1.8, max_value=45.5, value=7.5, step=0.1)
cholinesterase = st.sidebar.number_input('Cholinesterase (kU/L)', min_value=2.8, max_value=15.4, value=6.9, step=0.1)
cholesterol = st.sidebar.number_input('Cholesterol (mg/dL)', min_value=41.0, max_value=127.0, value=60.0, step=0.1)
gamma_glutamyl_transferase = st.sidebar.number_input('Gamma Glutamyl Transferase (IU/L)', min_value=7.0, max_value=185.0, value=12.1, step=0.1)
protein = st.sidebar.number_input('Protein (g/L)', min_value=53.0, max_value=86.5, value=70.0, step=0.1)

manual_input = pd.DataFrame([[albumin, alkaline_phosphatase, alanine_aminotransferase,
                              aspartate_aminotransferase, bilirubin, cholinesterase,
                              cholesterol, gamma_glutamyl_transferase, protein]],
                            columns=feature_names)

manual_scaled = pd.DataFrame(scaler.transform(manual_input), columns=feature_names)

if st.button("🔮 Predict Single Case"):
    manual_prediction = model.predict(manual_scaled)[0]
    manual_proba = model.predict_proba(manual_scaled)[0]

    st.write("🧾 Predicted Class:", f"**{class_map[manual_prediction]}**")

    # Probability table
    prob_df = pd.DataFrame({
        "Class": [class_map[i] for i in range(len(manual_proba))],
        "Probability": manual_proba
    })
    st.write("### Probability Distribution")
    st.table(prob_df)

    # Bar chart
    st.write("### 📊 Probability Bar Chart")
    fig, ax = plt.subplots()
    ax.bar(prob_df["Class"], prob_df["Probability"], color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_title("Class Probability Distribution")
    st.pyplot(fig)
