# app.py
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

# Load data
diabetes_df = pd.read_csv('diabetes.csv')
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# Load saved ANN model and scaler
model = load_model("diabetes_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# Define correct feature order
FEATURES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# Streamlit App
def app():
    # Header image
    img = Image.open("chandu vs1.jpg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title('🩺 Diabetes Prediction using Neural Network')

    # Sidebar Inputs
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Create DataFrame input
    input_df = pd.DataFrame([[preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]], columns=FEATURES)

    # Predict button
    if st.sidebar.button('🔍 Predict Diabetes'):
        # Scale input
        input_scaled = scaler.transform(input_df)

        # Model prediction
        proba = model.predict(input_scaled)[0][0]
        prediction = int(proba > 0.5)

        # Display probability
        st.write(f"**Predicted Probability of Diabetes:** {proba:.3f}")

        # Display Prediction
        st.subheader('Prediction Result:')
        if prediction == 1:
            st.warning('⚠️ This person has diabetes.')
        else:
            st.success('✅ This person does not have diabetes.')

    # Dataset info
    st.header('📊 Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('📈 Average Feature Values by Outcome')
    st.write(diabetes_mean_df)

# Run app
if __name__ == '__main__':
    app()
