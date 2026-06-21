import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model & scaler saat server start ─────────────────
print("Loading model...")
MODEL_PATH = "expense_predictor_v3"
model = tf.keras.models.load_model(MODEL_PATH)

# Load scaler jika ada
SCALER_PATH = "feature_scaler.pkl"
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
print("Model loaded successfully!")

# ── Urutan fitur HARUS sama persis dengan saat training ───
FEATURE_ORDER = [
    "monthly_income",
    "savings_rate",
    "budget_goal",
    "debt_to_income_ratio",
    "loan_payment",
    "investment_amount",
    "subscription_services",
    "emergency_fund",
    "transaction_count",
    "discretionary_spending",
    "essential_spending",
    "rent_or_mortgage",
    "financial_stress_level",
    "income_type_Freelance",
    "income_type_Mixed",
    "income_type_Salary",
    "financial_scenario_inflation",
    "financial_scenario_normal",
    "financial_scenario_recession",
]

# Fitur yang perlu di-scale (12 fitur kontinu)
SCALE_FEATURES = [
    "monthly_income", "savings_rate", "budget_goal",
    "debt_to_income_ratio", "loan_payment", "investment_amount",
    "subscription_services", "emergency_fund", "transaction_count",
    "discretionary_spending", "essential_spending", "rent_or_mortgage",
]

MAX_INCOME = 20_000_000  # untuk denormalisasi output


# ── Health check ───────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "SmartFinance AI Server berjalan 🚀",
        "model": MODEL_PATH,
        "total_features": len(FEATURE_ORDER),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Endpoint prediksi ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validasi semua fitur ada
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                "error": "Missing features",
                "missing_fields": missing
            }), 400

        # Susun input sesuai urutan
        raw_values = [float(data[f]) for f in FEATURE_ORDER]

        # Validasi one-hot encoding
        income_types = [
            data.get("income_type_Freelance", 0),
            data.get("income_type_Mixed", 0),
            data.get("income_type_Salary", 0),
        ]
        scenario_types = [
            data.get("financial_scenario_inflation", 0),
            data.get("financial_scenario_normal", 0),
            data.get("financial_scenario_recession", 0),
        ]
        if sum(income_types) != 1:
            return jsonify({
                "error": "Tepat satu income_type_* harus bernilai 1"
            }), 400
        if sum(scenario_types) != 1:
            return jsonify({
                "error": "Tepat satu financial_scenario_* harus bernilai 1"
            }), 400

        # Buat array input
        input_array = np.array([raw_values])

        # Scaling jika scaler tersedia
        if scaler:
            scale_indices = [FEATURE_ORDER.index(f) for f in SCALE_FEATURES]
            input_array[:, scale_indices] = scaler.transform(
                input_array[:, scale_indices]
            )

        # Prediksi
        prediction = model.predict(input_array, verbose=0)
        predicted_normalized = float(prediction[0][0])
        predicted_rupiah = predicted_normalized * MAX_INCOME

        return jsonify({
            "predicted_expense_normalized": round(predicted_normalized, 4),
            "predicted_expense_rupiah": round(predicted_rupiah, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)