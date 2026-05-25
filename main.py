import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import traceback

load_dotenv()

# --- 1. DEFINISI CUSTOM LAYER (Wajib di atas sebelum load_model) ---
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
        super().build(input_shape)
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

# --- 2. FUNGSI ALOKASI ---
def calculate_allocation(category, saving_ratio):
    if "Reksadana" in category:
        return {"reksadana": 70, "obligasi": 25, "saham": 5}
    elif "Saham" in category:
        saham_pct = int(np.clip(40 + (saving_ratio * 40), 40, 80))
        return {"reksadana": 10, "obligasi": 90 - saham_pct, "saham": saham_pct}
    else:
        return {"reksadana": 20, "obligasi": 60, "saham": 20}

# --- 3. INIT APP & LOADING ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model, scaler = None, None
print("--- MEMULAI PROSES LOADING ---")

try:
    # Pastikan file ini ada di root folder Hugging Face
    if os.path.exists('model.h5'):
        model = tf.keras.models.load_model(
            'model.h5',
            custom_objects={
                'FinancialAttentionLayer': FinancialAttentionLayer,
                'HybridFinancialLoss': HybridFinancialLoss
            },
            compile=False
        )
        print("✅ Model loaded!")
    else:
        print("⚠️ File model.h5 tidak ditemukan!")

    if os.path.exists('scaler_v2.pkl'):
        with open('scaler_v2.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded!")
    else:
        print("⚠️ File scaler_v2.pkl tidak ditemukan!")

except Exception as e:
    print("❌ GAGAL LOAD MODEL/SCALER")
    traceback.print_exc()

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

@app.get("/")
async def index():
    # Pastikan file HTML ini juga di-upload
    return FileResponse('smartbudget_dashboard.html')

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict")
async def predict(data: UserInput):
    try:
        # Preprocessing Dasar
        income_log = np.log1p(data.income)
        monthly_inc = data.income / 12
        exp_ratio = np.clip(data.total_expense / (monthly_inc + 1e-6), 0, 1)
        lifestyle_burden = np.clip(data.lifestyle_expense / (monthly_inc + 1e-6), 0, 1)
        debt_burden = data.loan_int_rate / 20.0
        credit_norm = (data.credit_score - 300) / 549.0

        # Inisialisasi Default (Agar tidak error jika ML gagal)
        recommendation = "Obligasi (Moderat)"
        ratio = 0.20

        # LOGIKA OVERRIDE (Rule-Based Prioritas)
        if exp_ratio < 0.2 and data.income > 50000000:
            recommendation, ratio = "Saham (Agresif)", 0.65
        elif exp_ratio > 0.8 or data.credit_score < 450:
            recommendation, ratio = "Reksadana (Konservatif)", 0.05
        else:
            # JALANKAN MODEL ML JIKA TERSEDIA
            if model is not None and scaler is not None:
                try:
                    feats = np.array([[income_log, exp_ratio, lifestyle_burden, debt_burden, credit_norm,
                            data.risk_status, data.family_size, data.education, data.age, data.budget_encoded]])
                    feats_s = scaler.transform(feats)
                    preds = model.predict(feats_s, verbose=0)

                    if isinstance(preds, dict):
                        ratio = float(preds['ratio_output'][0][0])
                        cat_idx = int(np.argmax(preds['cat_output'][0]))
                    else:
                        ratio = float(preds[0][0][0])
                        cat_idx = int(np.argmax(preds[1][0]))

                    labels = {0: "Reksadana (Konservatif)", 1: "Obligasi (Moderat)", 2: "Saham (Agresif)"}
                    recommendation = labels.get(cat_idx, "Obligasi (Moderat)")
                except Exception as e_ml:
                    print(f"⚠️ ML Prediction Fail, using fallback: {e_ml}")

        # PANGGIL GEMINI AI
        advice = "Analisis selesai. Fokuslah pada pengelolaan pengeluaran dan investasi rutin."
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                gem_model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Berikan saran finansial sangat singkat (maks 2 kalimat) untuk saving ratio {ratio:.1%} dan profil {recommendation}"
                resp = gem_model.generate_content(prompt)
                if resp and resp.text:
                    advice = resp.text.strip()
            except Exception as e_gem:
                print(f"⚠️ Gemini Fail: {e_gem}")

        return {
            "predicted_saving_ratio": float(round(ratio, 4)),
            "recommendation": str(recommendation),
            "asset_allocation": calculate_allocation(recommendation, ratio),
            "ai_advice": str(advice)
        }

    except Exception as e_final:
        print(f"🔥 CRASH TOTAL: {e_final}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada sistem analisis.")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face butuh port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)