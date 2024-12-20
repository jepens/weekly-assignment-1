# Import library yang diperlukan
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Fungsi untuk memuat model dan scaler
def load_models():
    """
    Memuat model dan scaler yang telah disimpan
    """
    with open('boston_house_lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)
    
    with open('boston_house_rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
        
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return lr_model, rf_model, scaler

# Fungsi untuk melakukan prediksi
def predict_price(model, scaler, features):
    """
    Melakukan prediksi harga berdasarkan input features
    """
    # Mengubah input menjadi array dan melakukan scaling
    features_scaled = scaler.transform([features])
    # Melakukan prediksi
    prediction = model.predict(features_scaled)
    return prediction[0]

# Main function
def main():
    # Header aplikasi
    st.title("🏠 Boston House Price Prediction - TIM NLP ALAN TURING")
    st.write("Sistem prediksi harga rumah menggunakan Machine Learning")
    
    # Sidebar untuk pemilihan model
    st.sidebar.header("Model Selection (Pilih Model)")
    model_choice = st.sidebar.radio(
        "Pilih Model Prediksi:",
        ["Linear Regression", "Random Forest"]
    )
    
    # Load models
    lr_model, rf_model, scaler = load_models()
    
    # Form input untuk fitur
    st.subheader("Input Parameters")
    
    # Buat 3 kolom untuk input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        crim = st.number_input('CRIM (Crime Rate)', min_value=0.0, format='%f')
        zn = st.number_input('ZN (Residential Land)', min_value=0.0, format='%f')
        indus = st.number_input('INDUS (Industry)', min_value=0.0, format='%f')
        chas = st.selectbox('CHAS (Charles River)', [0, 1])
        
    with col2:
        nox = st.number_input('NOX (Nitric Oxide)', min_value=0.0, format='%f')
        rm = st.number_input('RM (Rooms)', min_value=0.0, format='%f')
        age = st.number_input('AGE', min_value=0.0, format='%f')
        dis = st.number_input('DIS (Distance)', min_value=0.0, format='%f')
        
    with col3:
        rad = st.number_input('RAD (Highway)', min_value=0.0, format='%f')
        tax = st.number_input('TAX', min_value=0.0, format='%f')
        ptratio = st.number_input('PTRATIO', min_value=0.0, format='%f')
        black = st.number_input('BLACK', min_value=0.0, format='%f')
        lstat = st.number_input('LSTAT', min_value=0.0, format='%f')
    
    # Button untuk prediksi
    if st.button('Predict Price'):
        # Menyusun features
        features = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat]
        
        # Memilih model berdasarkan input user
        if model_choice == "Linear Regression":
            model = lr_model
            model_name = "Linear Regression"
        else:
            model = rf_model
            model_name = "Random Forest"
        
        # Melakukan prediksi
        prediction = predict_price(model, scaler, features)
        
        # Menampilkan hasil
        st.success(f"Predicted House Price (using {model_name}): ${prediction:,.2f}")
        
        # Menampilkan interpretasi
        st.subheader("Model Interpretation")
        if model_choice == "Random Forest":
            feature_importance = pd.DataFrame({
                'Feature': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'BLACK', 'LSTAT'],
                'Importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            st.bar_chart(data=feature_importance.set_index('Feature'))
            st.write("Feature importance menunjukkan kontribusi relatif setiap fitur dalam prediksi")

if __name__ == '__main__':
    main()
