# =====================================================
# FINAL MODEL + ALL GRAPHS + FORECAST (ERROR-FREE)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("water_dataset/water_leak_dataset.csv")

# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================

df["diameter"] = np.random.uniform(0.2, 1.0, len(df))

flow_map = {"Low": 1.5, "Medium": 3.0, "High": 5.0}
df["velocity"] = df["flow"].map(flow_map) + np.random.normal(0, 0.3, len(df))

# Interaction features
df["pressure_velocity"] = df["pressure"] * df["velocity"]
df["diameter_pressure"] = df["diameter"] * df["pressure"]

# Encode zone
le_zone = LabelEncoder()
df["zone"] = le_zone.fit_transform(df["zone"])

# =====================================================
# ⚠️ FIX FOR HEATMAP (ENCODE FLOW)
# =====================================================

flow_encode = {"Low": 1, "Medium": 2, "High": 3}
df["flow_encoded"] = df["flow"].map(flow_encode)

# =====================================================
# 3. FEATURES & TARGET
# =====================================================

X = df[[
    "zone", "pressure", "diameter", "velocity",
    "pressure_velocity", "diameter_pressure"
]].values

y = df["leak"].values

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =====================================================
# 4. ANN MODEL
# =====================================================

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

history = model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)

ann_prob = model.predict(X_test, verbose=0)
ann_pred = (ann_prob > 0.5).astype(int)
ann_acc = accuracy_score(y_test, ann_pred)

# =====================================================
# 5. SVM MODEL
# =====================================================

svm = SVC(probability=True, C=2)
svm.fit(X_train, y_train)

svm_prob = svm.predict_proba(X_test)[:, 1]
svm_pred = (svm_prob > 0.5).astype(int)
svm_acc = accuracy_score(y_test, svm_pred)

# =====================================================
# 6. HYBRID MODEL
# =====================================================

hybrid_prob = 0.6 * ann_prob.flatten() + 0.4 * svm_prob
hybrid_pred = (hybrid_prob > 0.5).astype(int)
hybrid_acc = accuracy_score(y_test, hybrid_pred)

print("\n✅ ANN Accuracy:", round(ann_acc * 100, 2), "%")
print("✅ SVM Accuracy:", round(svm_acc * 100, 2), "%")
print("✅ Hybrid Accuracy:", round(hybrid_acc * 100, 2), "%")

# =====================================================
# 7. ALL GRAPHS (PLACED BEFORE INPUT ✅)
# =====================================================

# Accuracy Comparison
plt.figure()
plt.bar(["ANN", "SVM", "Hybrid"], [ann_acc, svm_acc, hybrid_acc])
plt.title("Model Accuracy Comparison")
plt.show()

# ROC Curve
ann_fpr, ann_tpr, _ = roc_curve(y_test, ann_prob)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
hyb_fpr, hyb_tpr, _ = roc_curve(y_test, hybrid_prob)

plt.figure()
plt.plot(ann_fpr, ann_tpr, label="ANN")
plt.plot(svm_fpr, svm_tpr, label="SVM")
plt.plot(hyb_fpr, hyb_tpr, label="Hybrid")
plt.legend()
plt.title("ROC Curve")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, hybrid_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# Heatmap (FIXED)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure()
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Loss Graph
plt.figure()
plt.plot(history.history['loss'])
plt.title("ANN Training Loss")
plt.show()

# Distribution
plt.figure()
plt.hist(hybrid_prob, bins=20)
plt.title("Probability Distribution")
plt.show()

# Scatter plots
plt.figure()
plt.scatter(df["pressure"], df["leak"])
plt.title("Pressure vs Leak")
plt.show()

plt.figure()
plt.scatter(df["velocity"], df["leak"])
plt.title("Velocity vs Leak")
plt.show()

# Zone analysis
plt.figure()
df.groupby("zone")["leak"].mean().plot(kind='bar')
plt.title("Leak Rate by Zone")
plt.show()

# =====================================================
# 8. FORECAST FUNCTION
# =====================================================

def forecast_leak(zone_encoded, pressure, diameter, velocity, days=8):

    print("\n📅 8-Day Forecast:\n")

    probs = []

    for day in range(1, days + 1):

        pressure_day = pressure + np.random.normal(0, 2)
        velocity_day = velocity + np.random.normal(0, 0.2)

        degradation = 1 + (day * 0.01)
        pressure_day *= degradation
        velocity_day *= degradation

        pv = pressure_day * velocity_day
        dp = diameter * pressure_day

        data = np.array([[zone_encoded, pressure_day, diameter, velocity_day, pv, dp]])
        data = scaler.transform(data)

        prob = 0.6 * model.predict(data, verbose=0)[0][0] + \
               0.4 * svm.predict_proba(data)[0][1]

        probs.append(prob)

        status = "🚨 Leak Risk" if prob > 0.5 else "✅ Safe"
        print(f"Day {day}: {status} | {round(prob,3)}")

    # Forecast graph
    plt.figure()
    plt.plot(range(1, days+1), probs, marker='o')
    plt.title("Future Leak Forecast")
    plt.xlabel("Days")
    plt.ylabel("Probability")
    plt.show()

# =====================================================
# 9. USER INPUT
# =====================================================

def predict_leak():

    zone_input = input("\nZone (A/B/C/D): ").strip().upper()

    if zone_input in ["A", "B", "C", "D"]:
        zone_input = "Zone " + zone_input

    if zone_input not in le_zone.classes_:
        print("❌ Invalid Zone")
        return

    pressure = float(input("Pressure: "))
    diameter = float(input("Diameter: "))
    velocity = float(input("Velocity: "))

    zone_encoded = le_zone.transform([zone_input])[0]

    pv = pressure * velocity
    dp = diameter * pressure

    data = np.array([[zone_encoded, pressure, diameter, velocity, pv, dp]])
    data = scaler.transform(data)

    prob = 0.6 * model.predict(data, verbose=0)[0][0] + \
           0.4 * svm.predict_proba(data)[0][1]

    print("\nProbability:", round(prob,3))
    print("🚨 Leak" if prob > 0.5 else "✅ Safe")

    forecast_leak(zone_encoded, pressure, diameter, velocity)

# =====================================================
# RUN
# =====================================================

predict_leak()