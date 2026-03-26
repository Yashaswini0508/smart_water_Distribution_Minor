from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import History
import os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Global model objects ──────────────────────────────────────────────────────
ann_model  = None
svm_model  = None
rf_model   = None
scaler     = None
le_zone    = None
ZONE_CLASSES = []

hybrid_acc = None
ann_acc    = None
svm_acc    = None
rf_acc     = None

# Store eval data globally so /eval_data can serve it
_eval_data = {}

# ── Dataset generation (used when CSV not present) ────────────────────────────
def generate_dataset(n=1200):
    np.random.seed(42)
    zones    = np.random.choice(["Zone A", "Zone B", "Zone C", "Zone D"], n)
    pressure = np.random.uniform(20, 90, n)
    flow     = np.random.choice(["Low", "Medium", "High"], n)
    leak     = ((pressure > 60) | (flow == "High")).astype(int)
    leak     = np.where(np.random.random(n) < 0.08, 1 - leak, leak)
    return pd.DataFrame({"zone": zones, "pressure": pressure, "flow": flow, "leak": leak})

# ── Train on startup ──────────────────────────────────────────────────────────
def train_models():
    global ann_model, svm_model, rf_model, scaler, le_zone
    global hybrid_acc, ann_acc, svm_acc, rf_acc, ZONE_CLASSES, _eval_data

    csv_path = "water_dataset/water_leak_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {len(df)} rows")
    else:
        df = generate_dataset()
        print(f"✅ Generated synthetic dataset: {len(df)} rows")

    # ── Feature engineering ──────────────────────────────────────────────────
    np.random.seed(42)
    df["diameter"] = np.random.uniform(0.2, 1.0, len(df))

    flow_map = {"Low": 1.5, "Medium": 3.0, "High": 5.0}
    df["velocity"] = df["flow"].map(flow_map) + np.random.normal(0, 0.3, len(df))
    df["velocity"] = df["velocity"].clip(lower=0.1)

    df["pressure_velocity"] = df["pressure"] * df["velocity"]
    df["diameter_pressure"] = df["diameter"]  * df["pressure"]

    # ── BUG FIX: Keep original zone strings, encode separately ──────────────
    le_zone = LabelEncoder()
    df["zone_encoded"] = le_zone.fit_transform(df["zone"])   # fit on strings
    ZONE_CLASSES = list(le_zone.classes_)                    # ['Zone A', 'Zone B', ...]
    print(f"📌 Zone classes: {ZONE_CLASSES}")

    FEATURES = ["zone_encoded", "pressure", "diameter", "velocity",
                "pressure_velocity", "diameter_pressure"]

    X = df[FEATURES].values
    y = df["leak"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── ANN ──────────────────────────────────────────────────────────────────
    print("🧠 Training ANN...")
    hist = History()
    ann_model = Sequential([
        Dense(128, activation='relu', input_dim=X_scaled.shape[1]),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1,  activation='sigmoid')
    ])
    ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = ann_model.fit(X_train, y_train, epochs=40, batch_size=32,
                            verbose=0, validation_split=0.1)

    ann_prob_test = ann_model.predict(X_test, verbose=0).flatten()
    ann_pred_test = (ann_prob_test > 0.5).astype(int)
    ann_acc = round(accuracy_score(y_test, ann_pred_test) * 100, 2)
    print(f"  ANN Accuracy: {ann_acc}%")

    # ── SVM ──────────────────────────────────────────────────────────────────
    print("⚙️  Training SVM...")
    svm_model = SVC(kernel='rbf', probability=True, C=2, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    svm_prob_test = svm_model.predict_proba(X_test)[:, 1]
    svm_pred_test = (svm_prob_test > 0.5).astype(int)
    svm_acc = round(accuracy_score(y_test, svm_pred_test) * 100, 2)
    print(f"  SVM Accuracy: {svm_acc}%")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_prob_test = rf_model.predict_proba(X_test)[:, 1]
    rf_pred_test = (rf_prob_test > 0.5).astype(int)
    rf_acc = round(accuracy_score(y_test, rf_pred_test) * 100, 2)
    print(f"  RF Accuracy:  {rf_acc}%")

    # ── Hybrid (ANN 50% + SVM 30% + RF 20%) ──────────────────────────────────
    hybrid_prob_test = 0.5 * ann_prob_test + 0.3 * svm_prob_test + 0.2 * rf_prob_test
    hybrid_pred_test = (hybrid_prob_test > 0.5).astype(int)
    hybrid_acc = round(accuracy_score(y_test, hybrid_pred_test) * 100, 2)
    print(f"  Hybrid Acc:   {hybrid_acc}%")

    # ── Build eval data ───────────────────────────────────────────────────────
    def confusion(y_true, y_pred):
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def roc(y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        au = round(auc(fpr, tpr), 3)
        # downsample to 40 points
        idx = np.linspace(0, len(fpr)-1, 40, dtype=int)
        return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": au}

    # Feature importance from RF
    feat_names = FEATURES
    feat_imp   = rf_model.feature_importances_.tolist()

    safe_count = int((y == 0).sum())
    leak_count = int((y == 1).sum())

    _eval_data = {
        "train_loss": [round(v, 4) for v in history.history["loss"]],
        "val_loss":   [round(v, 4) for v in history.history["val_loss"]],
        "feat_names": feat_names,
        "feat_imp":   feat_imp,
        "ann_roc":    roc(y_test, ann_prob_test),
        "svm_roc":    roc(y_test, svm_prob_test),
        "rf_roc":     roc(y_test, rf_prob_test),
        "hybrid_roc": roc(y_test, hybrid_prob_test),
        "ann_cm":     confusion(y_test, ann_pred_test),
        "svm_cm":     confusion(y_test, svm_pred_test),
        "rf_cm":      confusion(y_test, rf_pred_test),
        "hybrid_cm":  confusion(y_test, hybrid_pred_test),
        "class_balance": {"safe": safe_count, "leak": leak_count}
    }

    print(f"\n✅ All models ready | Hybrid: {hybrid_acc}% | ANN: {ann_acc}% | SVM: {svm_acc}% | RF: {rf_acc}%")
    print("🌐 Starting Flask...\n")


# ── Helper: build scaled feature array ───────────────────────────────────────
def build_features(zone_str, pressure, diameter, velocity):
    """Return scaled feature vector for prediction."""
    zone_enc = int(le_zone.transform([zone_str])[0])
    pv = pressure * velocity
    dp = diameter * pressure
    raw = np.array([[zone_enc, pressure, diameter, velocity, pv, dp]])
    return scaler.transform(raw)


# ── Helper: parse zone string safely ─────────────────────────────────────────
def parse_zone(raw):
    s = str(raw).strip().upper()
    # "A" → "Zone A"
    if len(s) == 1 and s in "ABCD":
        return "Zone " + s
    # "ZONE A" → "Zone A"
    if s.startswith("ZONE ") and len(s) == 6 and s[-1] in "ABCD":
        return "Zone " + s[-1]
    # title-case fallback
    candidate = s.title()
    if candidate in ZONE_CLASSES:
        return candidate
    return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           zones=ZONE_CLASSES,
                           hybrid_acc=hybrid_acc,
                           ann_acc=ann_acc,
                           svm_acc=svm_acc,
                           rf_acc=rf_acc)          # ← BUG FIX: rf_acc now passed


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        # ── Zone ──────────────────────────────────────────────────────────────
        zone_str = parse_zone(data.get("zone", ""))
        if zone_str is None:
            return jsonify({"error": f"Invalid zone '{data.get('zone')}'. Valid: {ZONE_CLASSES}"}), 400

        # ── Numerics ──────────────────────────────────────────────────────────
        try:
            pressure    = float(data["pressure"])
            diameter    = float(data["diameter"])
            velocity    = float(data["velocity"])
            temperature = float(data.get("temperature", 25))
            age         = float(data.get("age", 10))
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid numeric field: {e}"}), 400

        velocity = max(velocity, 0.1)   # BUG FIX: clip velocity

        # ── Predict ───────────────────────────────────────────────────────────
        X_input = build_features(zone_str, pressure, diameter, velocity)
        ann_p   = float(ann_model.predict(X_input, verbose=0)[0][0])
        svm_p   = float(svm_model.predict_proba(X_input)[0][1])
        rf_p    = float(rf_model.predict_proba(X_input)[0][1])
        prob    = 0.5 * ann_p + 0.3 * svm_p + 0.2 * rf_p

        # ── 8-Day Forecast ────────────────────────────────────────────────────
        forecast = []
        base_temp = temperature
        for day in range(1, 9):
            factor   = 1 + day * 0.012
            p_d = max((pressure + np.random.normal(0, 2)) * factor, 10)
            v_d = max((velocity + np.random.normal(0, 0.2)) * factor, 0.1)
            t_d = base_temp + np.random.normal(0, 0.8) * day * 0.3

            fd   = build_features(zone_str, p_d, diameter, v_d)
            ann_fd  = float(ann_model.predict(fd, verbose=0)[0][0])
            svm_fd  = float(svm_model.predict_proba(fd)[0][1])
            rf_fd   = float(rf_model.predict_proba(fd)[0][1])
            fp      = 0.5 * ann_fd + 0.3 * svm_fd + 0.2 * rf_fd

            forecast.append({
                "day":         day,
                "probability": round(fp, 4),
                "pressure":    round(p_d, 2),
                "velocity":    round(v_d, 3),
                "temperature": round(t_d, 1),   # BUG FIX: include temperature
                "status":      "leak" if fp > 0.5 else "safe"
            })

        # ── Zone Comparison ───────────────────────────────────────────────────
        zone_sim = []
        for z in ZONE_CLASSES:
            zX   = build_features(z, pressure, diameter, velocity)
            z_ann = float(ann_model.predict(zX, verbose=0)[0][0])
            z_svm = float(svm_model.predict_proba(zX)[0][1])
            z_rf  = float(rf_model.predict_proba(zX)[0][1])
            z_prob = 0.5 * z_ann + 0.3 * z_svm + 0.2 * z_rf
            zone_sim.append({
                "zone":        z,
                "probability": round(z_prob, 4),
                "status":      "leak" if z_prob > 0.5 else "safe"
            })

        return jsonify({
            "probability": round(prob, 4),
            "ann_prob":    round(ann_p, 4),
            "svm_prob":    round(svm_p, 4),
            "rf_prob":     round(rf_p, 4),   # BUG FIX: rf_prob included
            "status":      "leak" if prob > 0.5 else "safe",
            "zone":        zone_str,
            "forecast":    forecast,          # BUG FIX: includes temperature
            "zone_sim":    zone_sim,          # BUG FIX: zone comparison data
            "model_acc":   {
                "ann":    ann_acc,
                "svm":    svm_acc,
                "rf":     rf_acc,
                "hybrid": hybrid_acc
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/eval_data")   # BUG FIX: this route was completely missing
def eval_data():
    return jsonify(_eval_data)


@app.route("/zones")
def zones():
    return jsonify({"zones": ZONE_CLASSES})


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "hybrid_acc": hybrid_acc,
        "ann_acc":    ann_acc,
        "svm_acc":    svm_acc,
        "rf_acc":     rf_acc
    })


if __name__ == "__main__":
    train_models()
    app.run(debug=True, port=5000)