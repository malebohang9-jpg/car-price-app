
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except:
        return None, None, None

model, scaler, label_encoders = load_model()

# App UI
st.title("🚗 Car Price Prediction App")
st.markdown("Predict your car's selling price using AI!")

with st.form("car_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Mahindra", "Tata", "Volkswagen"])
        vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
        km_driven = st.number_input("Kilometers Driven", 0, 500000, 50000)
        mileage = st.slider("Mileage (kmpl)", 5.0, 35.0, 20.0)
        
    with col2:
        engine = st.slider("Engine Size (cc)", 500, 5000, 1200)
        max_power = st.slider("Max Power (bhp)", 40, 500, 100)
        seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8])
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = {
        'vehicle_age': vehicle_age, 'km_driven': km_driven, 'mileage': mileage,
        'engine': engine, 'max_power': max_power, 'seats': seats,
        'brand': brand, 'fuel_type': fuel_type, 
        'transmission_type': transmission_type, 'seller_type': seller_type
    }
    
    input_df = pd.DataFrame([input_data])
    
    if model is not None:
        # Encode categorical variables
        for col in ['brand', 'fuel_type', 'transmission_type', 'seller_type']:
            if col in label_encoders:
                input_df[f'{col}_encoded'] = label_encoders[col].transform(input_df[col])
        
        # Select features and scale
        features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
                   'brand_encoded', 'fuel_type_encoded', 'transmission_type_encoded', 'seller_type_encoded']
        available_features = [f for f in features if f in input_df.columns]
        X_input = input_df[available_features]
        X_scaled = scaler.transform(X_input)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        st.success(f"### Predicted Price: ₹{prediction:,.2f}")
    else:
        # Fallback calculation
        base_price = 500000
        age_discount = vehicle_age * 20000
        prediction = base_price - age_discount
        prediction = max(prediction, 100000)
        st.success(f"### Estimated Price: ₹{prediction:,.2f}")

st.sidebar.info("Built with ❤️ using Streamlit & Machine Learning")
