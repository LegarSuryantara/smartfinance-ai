import os
import json
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
        self.units        = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Definisikan semua sub-layer di build()
        self.dense1  = keras.layers.Dense(self.units, activation='relu')
        self.dense2  = keras.layers.Dense(self.units)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.norm    = keras.layers.LayerNormalization()

        # Build semua sub-layer secara manual
        self.dense1.build(input_shape)
        self.dense2.build(input_shape)
        self.norm.build(input_shape)

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
MODEL_PATH   = "expense_predictor_v3.keras"
CONFIG_PATH  = os.path.join(MODEL_PATH, "config.json")
WEIGHTS_PATH = os.path.join(MODEL_PATH, "model.weights.h5")

model = None

try:
    # Load arsitektur dari config.json
    with open(CONFIG_PATH, "r") as f:
        config_json = json.load(f)

    model = keras.models.model_from_json(
        json.dumps(config_json),
        custom_objects={'ResidualBlock': ResidualBlock}
    )

    # Build model dulu sebelum load weights
    model.build(input_shape=(None, 19))

    # Load weights dengan skip_mismatch
    model.load_weights(WEIGHTS_PATH, skip_mismatch=True)

    print("✅ Model loaded successfully!")
    print(f"   Input shape : {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")

except Exception as e:
    print(f"❌ Model gagal dimuat: {e}")
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
        "status":       "ok",
        "message":      "SmartFinance AI Server 🚀",
        "model_loaded": model is not None,
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
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

        # Validasi semua fitur ada
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                "error":          "Missing features",
                "missing_fields": missing
            }), 400

        # Validasi one-hot income_type
        if sum([data.get("income_type_Freelance", 0),
                data.get("income_type_Mixed",     0),
                data.get("income_type_Salary",    0)]) != 1:
            return jsonify({
                "error": "Tepat satu income_type_* harus bernilai 1"
            }), 400

        # Validasi one-hot financial_scenario
        if sum([data.get("financial_scenario_inflation", 0),
                data.get("financial_scenario_normal",    0),
                data.get("financial_scenario_recession", 0)]) != 1:
            return jsonify({
                "error": "Tepat satu financial_scenario_* harus bernilai 1"
            }), 400

        # Susun input sesuai urutan
        input_array = np.array([[float(data[f]) for f in FEATURE_ORDER]])

        # Prediksi
        prediction           = model.predict(input_array, verbose=0)
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