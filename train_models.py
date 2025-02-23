import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

# ✅ 1️⃣ Train AI for BESS (Battery Energy Storage System)
def train_bess_model():
    df = pd.DataFrame({
        'Grid Load (MW)': [520, 500, 450, 480],
        'Battery SOC (%)': [85, 75, 60, 65],
        'Energy Price ($/kWh)': [0.07, 0.08, 0.10, 0.09],
        'Renewable Output (MW)': [320, 300, 280, 310],
        'Temperature (°C)': [33, 35, 30, 28],
        'Charging Efficiency (%)': [94, 92, 89, 90]
    })
    X = df.drop("Charging Efficiency (%)", axis=1)
    y = df["Charging Efficiency (%)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "bess_model.pkl")
    print("✅ BESS Model Trained & Saved!")

# ✅ 2️⃣ Train Fault Detection Model
def train_fault_model():
    df = pd.DataFrame({
        'Voltage (kV)': [220, 230, 225, 215],
        'Frequency (Hz)': [50.0, 49.5, 50.2, 49.0],
        'Transformer Load (MW)': [100, 120, 110, 130],
        'Vibration (g)': [0.02, 0.03, 0.02, 0.05],
        'Overcurrent Protection Triggered (Y/N)': ["No", "Yes", "No", "Yes"]
    })
    df['Overcurrent Protection Triggered (Y/N)'] = LabelEncoder().fit_transform(df['Overcurrent Protection Triggered (Y/N)'])
    
    X = df.drop("Overcurrent Protection Triggered (Y/N)", axis=1)
    y = df["Overcurrent Protection Triggered (Y/N)"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "fault_model.pkl")
    print("✅ Fault Detection Model Trained & Saved!")

# ✅ 3️⃣ Train Financial Model
def train_finance_model():
    df = pd.DataFrame({
        'Project Investment (₹ Cr)': [500, 750],
        'ROI (%)': [12, 9],
        'Interest Rate (%)': [6.5, 7.2],
        'Inflation (%)': [4.2, 5.1],
        'Financial Stability Index': [0.90, 0.75]
    })
    X = df.drop("Financial Stability Index", axis=1)
    y = df["Financial Stability Index"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "finance_model.pkl")
    print("✅ Financial Model Trained & Saved!")

# ✅ 4️⃣ Train Grid Load Prediction Model
def train_grid_model():
    df = pd.DataFrame({
        'Past Load (MW)': [500, 520],
        'Population Growth (%)': [2.1, 1.8],
        'Industrial Demand (MW)': [350, 370],
        'Residential Demand (MW)': [100, 110],
        'Power Grid Stress (%)': [80, 85]
    })
    X = df.drop("Power Grid Stress (%)", axis=1)
    y = df["Power Grid Stress (%)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "grid_model.pkl")
    print("✅ Grid Load Model Trained & Saved!")

# ✅ 5️⃣ Train Oil & Gas Model
def train_oil_gas_model():
    df = pd.DataFrame({
        'Oil Price ($/Barrel)': [80],
        'Refinery Efficiency (%)': [92],
        'Pipeline Pressure (PSI)': [3000],
        'CO₂ Emissions (kg CO₂/barrel)': [420]
    })
    X = df.drop("Refinery Efficiency (%)", axis=1)
    y = df["Refinery Efficiency (%)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "oil_gas_model.pkl")
    print("✅ Oil & Gas Model Trained & Saved!")

# ✅ 6️⃣ Train Renewable Energy Model
def train_renewable_model():
    df = pd.DataFrame({
        'Wind Speed (m/s)': [12],
        'Solar Irradiation (kWh/m²)': [5.5],
        'Battery Storage Utilization (%)': [85],
        'Grid Integration Rate (%)': [92]
    })
    X = df.drop("Grid Integration Rate (%)", axis=1)
    y = df["Grid Integration Rate (%)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "renewable_model.pkl")
    print("✅ Renewable Model Trained & Saved!")

# ✅ 7️⃣ Train Space & Aerospace Model
def train_space_model():
    df = pd.DataFrame({
        'Fuel Efficiency in Vacuum (ISP)': [350],
        'Satellite Orbit Stability (%)': [98],
        'Radiation Resistance (%)': [92],
        'Rocket Thrust (kN)': [3000]
    })
    X = df.drop("Satellite Orbit Stability (%)", axis=1)
    y = df["Satellite Orbit Stability (%)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "space_model.pkl")
    print("✅ Space Model Trained & Saved!")

# ✅ 8️⃣ Train Telecom AI Model
def train_telecom_model():
    df = pd.DataFrame({
        '5G Spectrum Usage (%)': [75],
        'Signal Strength (dBm)': [-60],
        'Network Latency (ms)': [20],
        'Active Users': [10000],
        'Data Traffic (TB)': [1.2]
    })
    X = df.drop("Network Latency (ms)", axis=1)
    y = df["Network Latency (ms)"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "telecom_model.pkl")
    print("✅ Telecom Model Trained & Saved!")

# 🚀 **Train All Models**
if __name__ == "__main__":
    train_bess_model()
    train_fault_model()
    train_finance_model()
    train_grid_model()
    train_oil_gas_model()
    train_renewable_model()
    train_space_model()
    train_telecom_model()
    print("🎯 All Models Trained & Saved Successfully!")
