import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Custom Layer ───────────────────────────────────────────
@keras.utils.register_keras_serializable()
class ResidualBlock(keras.layers.Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1  = keras.layers.Dense(units, activation='relu')
        self.dense2  = keras.layers.Dense(units)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.norm    = keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = x + inputs
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units':        self.units,
            'dropout_rate': self.dropout_rate,
        })
        return config

# ── Load model ─────────────────────────────────────────────
print("Loading model...")
MODEL_PATH = "expense_predictor_v3.keras"

try:
    # Coba load normal dulu
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'ResidualBlock': ResidualBlock},
        compile=False,
        safe_mode=False,
    )
    print("✅ Model loaded successfully!")

except Exception as e1:
    print(f"⚠️ Load normal gagal: {e1}")
    print("🔄 Mencoba load dengan skip_mismatch...")

    try:
        # Load hanya arsitektur lalu load weights terpisah
        import json, h5py

        config_path = os.path.join(MODEL_PATH, "config.json")
        weights_path = os.path.join(MODEL_PATH, "model.weights.h5")

        with open(config_path, "r") as f:
            config_json = f.read()

        model = keras.models.model_from_json(
            config_json,
            custom_objects={'ResidualBlock': ResidualBlock}
        )

        # Load weights dengan skip_mismatch=True
        model.load_weights(weights_path, skip_mismatch=True)
        print("✅ Model loaded dengan skip_mismatch!")

    except Exception as e2:
        print(f"❌ Semua metode load gagal: {e2}")
        model = None

# ── Konfigurasi fitur ──────────────────────────────────────
FEATURE_ORDER = [
    "monthly_income", "savings_rate", "budget_goal",
    "debt_to_income_ratio", "loan_payment", "investment_amount",
    "subscription_services", "emergency_fund", "transaction_count",
    "discretionary_spending", "essential_spending", "rent_or_mortgage",
    "financial_stress_level",
    "income_type_Freelance", "income_type_Mixed", "income_type_Salary",
    "financial_scenario_inflation", "financial_scenario_normal",
    "financial_scenario_recession",
]

MAX_INCOME = 20_000_000

# ── Routes ─────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "SmartFinance AI Server 🚀",
        "model_loaded": model is not None,
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model belum berhasil dimuat. Hubungi tim AI."
        }), 503

    try:
        data = request.get_json()

        # Validasi fitur
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                "error": "Missing features",
                "missing_fields": missing
            }), 400

        # Validasi one-hot
        if sum([data.get("income_type_Freelance", 0),
                data.get("income_type_Mixed", 0),
                data.get("income_type_Salary", 0)]) != 1:
            return jsonify({
                "error": "Tepat satu income_type_* harus bernilai 1"
            }), 400

        if sum([data.get("financial_scenario_inflation", 0),
                data.get("financial_scenario_normal", 0),
                data.get("financial_scenario_recession", 0)]) != 1:
            return jsonify({
                "error": "Tepat satu financial_scenario_* harus bernilai 1"
            }), 400

        # Prediksi
        input_array = np.array([[float(data[f]) for f in FEATURE_ORDER]])
        prediction  = model.predict(input_array, verbose=0)
        predicted_normalized = float(prediction[0][0])
        predicted_rupiah     = predicted_normalized * MAX_INCOME

        return jsonify({
            "predicted_expense_normalized": round(predicted_normalized, 4),
            "predicted_expense_rupiah":     round(predicted_rupiah, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)