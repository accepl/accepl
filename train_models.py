import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error

# ✅ Ensure 'models/' directory exists
if not os.path.exists("models"):
    os.makedirs("models")

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_scaled, y)

    joblib.dump(model, "models/bess_model.pkl")
    joblib.dump(scaler, "models/bess_scaler.pkl")
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

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/fault_model.pkl")
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

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/finance_model.pkl")
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

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/grid_model.pkl")
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

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/oil_gas_model.pkl")
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

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/renewable_model.pkl")
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

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/space_model.pkl")
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

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/telecom_model.pkl")
    print("✅ Telecom Model Trained & Saved!")

# ✅ 9️⃣ Train Maintenance Model
def train_maintenance_model():
    df = pd.DataFrame({
        'Equipment Age (Years)': [5, 10, 15],
        'Failure Rate (%)': [2, 5, 8],
        'Maintenance Cost ($)': [1000, 2500, 5000]
    })
    X = df.drop("Maintenance Cost ($)", axis=1)
    y = df["Maintenance Cost ($)"]

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/maintenance_model.pkl")
    print("✅ Maintenance Model Trained & Saved!")

# ✅ 🔟 Train Military AI Model
def train_military_model():
    df = pd.DataFrame({
        'Sensor Accuracy (%)': [90, 95],
        'Threat Level': [1, 3],
        'Response Time (ms)': [200, 150]
    })
    X = df.drop("Response Time (ms)", axis=1)
    y = df["Response Time (ms)"]

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "models/military_model.pkl")
    print("✅ Military Model Trained & Saved!")

# ✅ 1️⃣1️⃣ Train Loss Model
def train_loss_model():
    df = pd.DataFrame({'Power Loss (MW)': [5, 10, 15], 'Grid Load (MW)': [200, 400, 600]})
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[['Grid Load (MW)']], df['Power Loss (MW)'])
    joblib.dump(model, "models/loss_model.pkl")
    print("✅ Loss Model Trained & Saved!")

# 🚀 Train all 11 models
if __name__ == "__main__":
    train_loss_model()
