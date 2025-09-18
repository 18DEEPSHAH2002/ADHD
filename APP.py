import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import joblib

# -----------------------------
# Load the trained model & encoder
# -----------------------------
# If you saved as .h5
model = load_model("best_model.h5")

# If you used LabelEncoder earlier, save it and load here
label_encoder = joblib.load("label_encoder.pkl")  # you must save it after training

# -----------------------------
# App Title
# -----------------------------
st.title("🧠 ADHD Prediction App")
st.write("Fill in the questionnaire and click **Predict** to see the ADHD type.")

# -----------------------------
# Define input fields
# -----------------------------
features = [
    "Fails_attention_to_details",
    "Difficulty_sustaining_attention",
    "Does_not_listen",
    "Fails_to_finish_tasks",
    "Disorganized",
    "Avoids_mental_effort",
    "Loses_things",
    "Easily_distracted",
    "Forgetful",
    "Fidgets_or_squirms",
    "Leaves_seat",
    "Runs_or_restless",
    "Difficulty_playing_quietly",
    "On_the_go",
    "Talks_excessively",
    "Blurts_out_answers",
    "Difficulty_waiting_turn",
    "Interrupts_or_intrudes"
]

user_input = []
st.subheader("Enter symptom ratings (0–3)")
for f in features:
    val = st.slider(f, min_value=0, max_value=3, value=0)
    user_input.append(val)

# Convert to numpy array
input_data = np.array(user_input).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"### ✅ Predicted ADHD Type: **{predicted_label}**")
