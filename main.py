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

# --- 1. DEFINISI CUSTOM KOMPONEN ---
@tf.keras.utils.register_keras_serializable()
class FinancialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
    def call(self, inputs):
        attn_weights = tf.nn.softmax(tf.matmul(inputs, self.w) + self.b)
        return inputs * attn_weights
    def get_config(self):
        return {**super().get_config(), "units": self.units}

# --- 2. INIT APP & LOADING ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model, scaler = None, None

def load_all_artifacts():
    global model, scaler
    try:
        # Load Scaler
        if os.path.exists('scaler_v2.pkl'):
            scaler = pickle.load(open('scaler_v2.pkl', 'rb'))
            print("✅ Scaler Loaded")

        # LOAD MODEL DENGAN TRIK COMPILE=FALSE (Menghindari error quantization_config)
        model_path = 'model_final.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'FinancialAttentionLayer': FinancialAttentionLayer},
                compile=False # KUNCINYA DISINI: Lewati konfigurasi optimizer yang rusak
            )
            print("✅ Model ML Loaded (Safe Mode)")
    except Exception as e:
        print(f"❌ Gagal Load: {e}")

load_all_artifacts()

# --- 3. LOGIKA BISNIS ---
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

def calculate_allocation(category, ratio):
    if "Reksadana" in category: return {"reksadana": 70, "obligasi": 25, "saham": 5}
    if "Saham" in category: return {"reksadana": 10, "obligasi": 20, "saham": 70}
    return {"reksadana": 20, "obligasi": 60, "saham": 20}

@app.get("/")
async def index():
    return FileResponse('smartbudget_dashboard.html')

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict")
async def predict(data: UserInput):
    try:
        # Preprocessing
        inc_monthly = data.income / 12
        exp_ratio = np.clip(data.total_expense / (inc_monthly + 1e-6), 0, 1)
        inc_log = np.log1p(data.income)
        
        # Default Fallback Values
        ratio = 0.15
        recommendation = "Obligasi (Moderat)"

        # JALANKAN ML JIKA BERHASIL LOAD
        if model and scaler:
            try:
                # Siapkan 10 fitur sesuai urutan train.py
                feats = np.array([[
                    inc_log, exp_ratio, 
                    (data.lifestyle_expense / (inc_monthly + 1e-6)),
                    (data.loan_int_rate / 20.0),
                    ((data.credit_score - 300) / 550.0),
                    data.risk_status, data.family_size, data.education, data.age, data.budget_encoded
                ]])
                feats_s = scaler.transform(feats)
                preds = model.predict(feats_s, verbose=0)
                
                # Ambil hasil (Handling Keras 3 format)
                ratio = float(preds['ratio_output'][0][0])
                cat_idx = int(np.argmax(preds['cat_output'][0]))
                labels = {0: "Reksadana (Konservatif)", 1: "Obligasi (Moderat)", 2: "Saham (Agresif)"}
                recommendation = labels.get(cat_idx, "Obligasi (Moderat)")
            except:
                pass # Balik ke fallback jika prediksi crash

        # PANGGIL GEMINI (Model name fix)
        advice = "Analisis selesai. Fokuslah pada tabungan rutin."
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Gunakan nama model yang paling update
                gem_model = genai.GenerativeModel('gemini-1.5-flash') 
                resp = gem_model.generate_content(f"Saran keuangan singkat 1 kalimat untuk saving ratio {ratio:.1%} dan profil {recommendation}")
                if resp.text: advice = resp.text.strip()
            except:
                pass

        return {
            "predicted_saving_ratio": float(np.clip(ratio, 0, 1)),
            "recommendation": recommendation,
            "asset_allocation": calculate_allocation(recommendation, ratio),
            "ai_advice": advice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))