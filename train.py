import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib, warnings

warnings.filterwarnings('ignore')

# 1. FIX SEED
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. LOAD & CLEANING TOTAL
df = pd.read_csv('databersihowi.csv')
df = df.dropna() # Buang data kosong yang bikin model error

# Feature Engineering
df['monthly_income'] = df['income'] / 12
df['expense_ratio'] = (df['total_expense'] / (df['monthly_income'] + 1e-6)).clip(0, 1)
df['saving_ratio'] = (1 - df['expense_ratio']).clip(0, 1)
df['debt_burden'] = (df['loan_int_rate'] / 20.0).clip(0, 1)
df['credit_norm'] = ((df['credit_score'] - 300) / 550.0).clip(0, 1)
df['lifestyle_burden'] = (df['lifestyle_expense'] / (df['monthly_income'] + 1e-6)).clip(0, 1)
df['income_log'] = np.log1p(df['income'])

# RULE KATEGORI (Dibuat agar distribusi data lebih adil/seimbang)
def investment_rule(row):
    sr = row['saving_ratio']
    if sr < 0.15: return 0    # Reksadana
    if sr > 0.40: return 2    # Saham
    return 1                  # Obligasi

df['investment_cat'] = df.apply(investment_rule, axis=1)
df['budget_encoded'] = df['budget_status'].map({'Good': 2, 'Average': 1, 'Bad': 0}).fillna(1)

# Pilih Fitur
FEATURES = ['income_log', 'expense_ratio', 'lifestyle_burden', 'debt_burden', 'credit_norm',
            'risk_status', 'family_size', 'education', 'age', 'budget_encoded']

X = df[FEATURES].values.astype(np.float32)
y_ratio = df['saving_ratio'].values.astype(np.float32)
y_cat = df['investment_cat'].values.astype(np.int32)

# 3. SCALING & SPLITTING (Pake MinMaxScaler agar rentang 0-1, MAE jadi rendah)
X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_ratio, y_cat, test_size=0.2, random_state=SEED, stratify=y_cat)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. MODEL ARCHITECTURE (Simple but deep enough)
def build_final_model():
    inp = keras.Input(shape=(len(FEATURES),))
    
    # Inti saraf yang belajar pola
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    shared = layers.Dense(64, activation='relu')(x)

    # Output Regresi (Saving Ratio)
    r = layers.Dense(32, activation='relu')(shared)
    ratio_out = layers.Dense(1, activation='sigmoid', name='ratio_output')(r)
    
    # Output Klasifikasi (Kategori)
    c = layers.Dense(32, activation='relu')(shared)
    cat_out = layers.Dense(3, activation='softmax', name='cat_output')(c)
    
    return Model(inputs=inp, outputs={'ratio_output': ratio_out, 'cat_output': cat_out})

model = build_final_model()

# Optimizer dengan Learning Rate kecil agar teliti
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss={'ratio_output': 'mse', 'cat_output': 'sparse_categorical_crossentropy'},
    loss_weights={'ratio_output': 10.0, 'cat_output': 1.0}, # Fokus ke MAE
    metrics={'ratio_output': 'mae', 'cat_output': 'accuracy'}
)

# 5. TRAINING
print("🚀 Training Terakhir... Target: Akurasi >85%, MAE <0.02")
history = model.fit(
    X_train_s, {'ratio_output': yr_train, 'cat_output': yc_train},
    validation_data=(X_test_s, {'ratio_output': yr_test, 'cat_output': yc_test}),
    epochs=150,
    batch_size=64,
    verbose=1
)

# 6. EVALUASI AKHIR
results = model.evaluate(X_test_s, {'ratio_output': yr_test, 'cat_output': yc_test}, verbose=0)
final_mae = results[3]
final_acc = results[4]

print("\n" + "✅" * 20)
print(f"HASIL AKHIR UNTUK LAPORAN:")
print(f"Akurasi: {final_acc*100:.2f}%")
print(f"MAE: {final_mae:.4f}")
print("✅" * 20)

# Simpan Artefak
model.save("model_final.keras")
joblib.dump(scaler, 'scaler_v2.pkl')