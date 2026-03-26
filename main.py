# =====================================================
# WATER LEAKAGE DETECTION - COMPLETE PIPELINE
# Dataset Generation + Save + Load + ANN Training
# =====================================================

import pandas as pd
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# =====================================================
# 1. CREATE DATASET
# =====================================================

def generate_dataset():
    folder_path = "water_dataset"
    os.makedirs(folder_path, exist_ok=True)

    num_samples = 4000
    zones = ["Zone A", "Zone B", "Zone C", "Zone D"]

    def generate_sensor_data():
        pressure = round(random.uniform(15, 60), 2)
        flow = random.choice(["Low", "Medium", "High"])
        vibration = round(random.uniform(0, 1), 2)
        return pressure, flow, vibration

    def assign_leak(pressure, flow, vibration):
        if pressure < 30 and vibration > 0.6:
            return 1
        elif pressure < 40 and flow == "High":
            return 1
        else:
            return 0

    def assign_severity(pressure, vibration, leak):
        if leak == 0:
            return "None"
        elif pressure < 25 and vibration > 0.7:
            return "High"
        elif pressure < 35:
            return "Medium"
        else:
            return "Low"

    data = []

    for _ in range(num_samples):
        zone = random.choice(zones)
        pressure, flow, vibration = generate_sensor_data()
        leak = assign_leak(pressure, flow, vibration)
        severity = assign_severity(pressure, vibration, leak)

        data.append([zone, pressure, flow, vibration, leak, severity])

    df = pd.DataFrame(data, columns=[
        "zone", "pressure", "flow", "vibration", "leak", "severity"
    ])

    file_path = os.path.join(folder_path, "water_leak_dataset.csv")
    df.to_csv(file_path, index=False)

    print("✅ Dataset saved at:", file_path)
    return file_path


# =====================================================
# 2. LOAD & PREPROCESS DATA
# =====================================================

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    print("\n🔍 Sample Data:\n", df.head())

    # Encode categorical features
    le_zone = LabelEncoder()
    le_flow = LabelEncoder()
    le_severity = LabelEncoder()

    df["zone"] = le_zone.fit_transform(df["zone"])
    df["flow"] = le_flow.fit_transform(df["flow"])
    df["severity"] = le_severity.fit_transform(df["severity"])

    # Features & Target
    X = df[["zone", "pressure", "flow", "vibration"]].values
    y = to_categorical(df["leak"])

    return X, y


# =====================================================
# 3. BUILD ANN MODEL
# =====================================================

def build_model(input_dim):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =====================================================
# 4. TRAIN MODEL
# =====================================================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = build_model(X.shape[1])

    print("\n🚀 Training Model...\n")

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    loss, acc = model.evaluate(X_test, y_test)

    print("\n✅ Test Accuracy:", acc)

    return model


# =====================================================
# 5. MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    # Step 1: Generate dataset
    file_path = generate_dataset()

    # Step 2: Preprocess
    X, y = preprocess_data(file_path)

    # Step 3: Train ANN
    model = train_model(X, y)

    # Step 4: Save model
    model.save("water_leak_model.h5")

    print("\n🎉 Model saved as water_leak_model.h5")