import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Custom Layer — WAJIB didefinisikan sebelum load_model ──
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

    def build(self, input_shape):
        # Build all sublayers with proper shapes
        self.dense1.build(input_shape)
        self.dense2.build((input_shape[0], self.units))
        self.dropout.build((input_shape[0], self.units))
        self.norm.build((input_shape[0], self.units))
        super().build(input_shape)

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
MODEL_PATH = "expense_predictor_v3"
model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={'ResidualBlock': ResidualBlock}
)
print("Model loaded successfully!")

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

# ── Health check ───────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "SmartFinance AI Server berjalan 🚀",
        "model": MODEL_PATH,
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

        # Validasi one-hot encoding
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

        # Susun input
        input_array = np.array([[float(data[f]) for f in FEATURE_ORDER]])

        # Prediksi
        prediction = model.predict(input_array, verbose=0)
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