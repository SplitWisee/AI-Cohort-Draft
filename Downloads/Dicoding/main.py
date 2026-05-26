import tensorflow as tf
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv 

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY tidak ditemukan di file .env!")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. DEFINISI CUSTOM LAYER (Tetap Sama) ---
@tf.keras.utils.register_keras_serializable()
class FinancialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, temperature=1.0, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units, self.temperature, self.dropout_rate = units, temperature, dropout_rate
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        n_feat = input_shape[-1]
        self.attn_w = self.add_weight(name='attn_w', shape=(n_feat, n_feat), initializer='glorot_uniform')
        self.attn_b = self.add_weight(name='attn_b', shape=(n_feat,), initializer='zeros')
        self.proj_w = self.add_weight(name='proj_w', shape=(n_feat, self.units), initializer='glorot_uniform')
        self.proj_b = self.add_weight(name='proj_b', shape=(self.units,), initializer='zeros')

    def call(self, inputs, training=False):
        scores = (tf.matmul(inputs, self.attn_w) + self.attn_b) / self.temperature
        weights = tf.nn.softmax(scores, axis=-1)
        attended = inputs * weights
        out = tf.nn.swish(tf.matmul(attended, self.proj_w) + self.proj_b)
        return self.dropout(out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "temperature": self.temperature, "dropout_rate": self.dropout_rate})
        return config

@tf.keras.utils.register_keras_serializable()
class HybridFinancialLoss(tf.keras.losses.Loss):
    def __init__(self, risk_threshold=0.35, risk_weight=2.5, delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.risk_threshold, self.risk_weight, self.delta = risk_threshold, risk_weight, delta

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
        err = y_true - y_pred
        huber = tf.where(tf.abs(err) <= self.delta, 0.5 * tf.square(err), self.delta * (tf.abs(err) - 0.5 * self.delta))
        weights = tf.where(y_true < self.risk_threshold, tf.ones_like(y_true) * self.risk_weight, tf.ones_like(y_true))
        return tf.reduce_mean(weights * huber)

    def get_config(self):
        config = super().get_config()
        config.update({"risk_threshold": self.risk_threshold, "risk_weight": self.risk_weight, "delta": self.delta})
        return config

# --- 3. INISIALISASI ---
app = FastAPI()
model = None
scaler = None

try:
    model = tf.keras.models.load_model(
        'best_robo_advisor_v2.keras', 
        custom_objects={'FinancialAttentionLayer': FinancialAttentionLayer, 'HybridFinancialLoss': HybridFinancialLoss},
        compile=False 
    )
    with open('scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model, Scaler, & Gemini Ready!")
except Exception as e:
    print(f"❌ Error: {e}")

class UserInput(BaseModel):
    income: float
    total_expense: float
    lifestyle_expense: float
    loan_int_rate: float
    credit_score: float
    risk_status: int
    family_size: int
    education: int
    age: int
    budget_encoded: int

async def get_gemini_advice(saving_ratio, category):
    prompt = f"""
    Kamu adalah pakar keuangan pintar.
    Hasil analisis: Saving Ratio {saving_ratio:.1%}, Rekomendasi: {category}.
    Berikan 2 kalimat saran singkat dan ramah dalam Bahasa Indonesia.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Tetap semangat menabung dan atur pengeluaranmu dengan bijak!"

# --- 5. ENDPOINT PREDICT ---
@app.post("/predict")
async def predict(data: UserInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Server not ready")
        
    try:
        # Preprocessing Dasar
        income_log = np.log1p(data.income)
        monthly_inc = data.income / 12
        exp_ratio = np.clip(data.total_expense / (monthly_inc + 1e-6), 0, 1)
        lifestyle_burden = np.clip(data.lifestyle_expense / (monthly_inc + 1e-6), 0, 1)
        debt_burden = data.loan_int_rate / 20.0
        credit_norm = (data.credit_score - 300) / 549.0

        # LOGIKA PENYELAMAT (Agar hasil bervariasi)
        if exp_ratio < 0.2 and data.income > 10000000:
            recommendation = "Saham (Agresif)"
            ratio = 0.45
        elif exp_ratio > 0.7 or data.credit_score < 500:
            recommendation = "Reksadana (Konservatif)"
            ratio = 0.05
        else:
            # Jika data moderat, baru pakai Model AI
            features = np.array([[
                income_log, exp_ratio, lifestyle_burden, debt_burden, credit_norm,
                data.risk_status, data.family_size, data.education, data.age, data.budget_encoded
            ]], dtype=np.float32)
            
            features_s = scaler.transform(features)
            preds = model.predict(features_s, verbose=0)
            
            ratio = float(preds['ratio_output'][0][0])
            cat_idx = int(np.argmax(preds['cat_output'][0]))
            labels = {0: "Reksadana", 1: "Obligasi", 2: "Saham"}
            recommendation = labels[cat_idx]

        # Panggil Gemini dengan hasil final (bisa dari Rule atau dari Model)
        ai_advice = await get_gemini_advice(ratio, recommendation)

        return {
            "predicted_saving_ratio": round(ratio, 4),
            "recommendation": recommendation,
            "ai_advice": ai_advice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))