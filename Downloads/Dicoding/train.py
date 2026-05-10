import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle, json, warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {len(tf.config.list_physical_devices("GPU")) > 0}')


2232
df = pd.read_csv('databersihowi.csv')

print(f'Shape: {df.shape}')
print(df.head(3).to_string())

print('='*65)
print(' DIAGNOSIS: Struktur Kolom Dataset')
print('='*65)

print(f"\nincome range    : {df['income'].min():,} – {df['income'].max():,}  (TAHUNAN)")
print(f"lifestyle_exp   : {df['lifestyle_expense'].min()} – {df['lifestyle_expense'].max()}  (BULANAN)")
print(f"total_expense   : {df['total_expense'].min()} – {df['total_expense'].max()}  (BULANAN)")
print(f"savings         : computed = income - total_expense (unit campur → TIDAK VALID sebagai rasio!)")


buggy_ratio = df['savings'] / df['income']
print(f"\n[BUG v1] savings/income → mean={buggy_ratio.mean():.3f}, min={buggy_ratio.min():.3f} — SEMUA >70%")

df['monthly_income']   = df['income'] / 12
df['expense_ratio']    = (df['total_expense'] / df['monthly_income']).clip(0, 1)
df['saving_ratio']     = (1 - df['expense_ratio']).clip(0, 1)

print(f"\n[FIX v2] expense_ratio = total_expense / (income/12)")
print(f"  saving_ratio mean  : {df['saving_ratio'].mean():.3f}")
print(f"  saving_ratio range : {df['saving_ratio'].min():.3f} – {df['saving_ratio'].max():.3f}")

# Distribusi bucket
print("\nDistribusi Saving Ratio (FIX):")
buckets = [0, 0.10, 0.15, 0.35, 0.50, 1.01]
labels  = ['<10% (Konservatif)', '10-15% (Border)', '15-35% (Moderat)', '35-50% (Agresif)', '>50%']
for i, lbl in enumerate(labels):
    mask = (df['saving_ratio'] >= buckets[i]) & (df['saving_ratio'] < buckets[i+1])
    cnt  = mask.sum()
    print(f"  {lbl}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

print("\n Diagnosis  — target v2 menggunakan expense_ratio & budget_status")

df['monthly_income']    = df['income'] / 12
df['expense_ratio']     = (df['total_expense'] / df['monthly_income']).clip(0, 1)
df['saving_ratio']      = 1 - df['expense_ratio']                        # TARGET REGRESI
df['debt_burden']       = df['loan_int_rate'] / 20.0                     # normalisasi 0-1
df['credit_norm']       = (df['credit_score'] - 300) / 549.0             # normalisasi 300-849
df['lifestyle_burden']  = (df['lifestyle_expense'] / df['monthly_income']).clip(0, 1)
df['income_log']        = np.log1p(df['income'])                         # log transform skewed

# Target 1: Investment Category (Rule-Based Ground Truth)
def investment_rule(row):
    """
    Kategorisasi investasi berdasarkan aturan bisnis fintech:
    Menggunakan TIGA sinyal: saving_ratio, loan_int_rate, credit_score
    Sesuai spesifikasi Robo-Advisor:
      Reksadana  → saving < 15% OR debt > 12% OR credit < 600
      Saham      → saving > 35% AND income tinggi AND lifestyle terkontrol
      Obligasi   → sisanya (moderat)
    """
    sr   = row['saving_ratio']
    debt = row['loan_int_rate']
    cs   = row['credit_score']
    li   = row['lifestyle_burden']
    inc  = row['monthly_income']

    # Konservatif
    if sr < 0.15 or debt > 12 or cs < 600:
        return 0  # Reksadana
    # Agresif
    elif sr > 0.35 and inc > 3000 and li < 0.4:
        return 2  # Saham
    # Moderat
    else:
        return 1  # Obligasi

df['investment_cat'] = df.apply(investment_rule, axis=1)

budget_map = {'Good': 2, 'Average': 1, 'Bad': 0}
df['budget_encoded'] = df['budget_status'].map(budget_map)

# Distribusi Target
inv_names = {0: 'Reksadana (Konservatif)', 1: 'Obligasi (Moderat)', 2: 'Saham (Agresif)'}
print('Investment Category Distribution (v2):')
for k, v in df['investment_cat'].value_counts().sort_index().items():
    print(f'  {inv_names[k]}: {v:,} ({v/len(df)*100:.1f}%)')

print(f'\nBudget Status Distribution:')
print(df['budget_status'].value_counts().to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(' Distribusi Target v2', fontsize=13, fontweight='bold')

axes[0].hist(df['saving_ratio'], bins=50, color='#2196F3', edgecolor='white')
axes[0].set_title('Saving Ratio (FIX)'); axes[0].set_xlabel('Ratio')

counts = df['investment_cat'].value_counts().sort_index()
axes[1].bar([inv_names[i] for i in counts.index], counts.values,
            color=['#F44336','#FF9800','#4CAF50'])
axes[1].set_title('Investment Category'); axes[1].tick_params(axis='x', rotation=15)

bs = df['budget_status'].value_counts()
axes[2].pie(bs, labels=bs.index, autopct='%1.1f%%',
            colors=['#4CAF50','#FF9800','#F44336'])
axes[2].set_title('Budget Status')

plt.tight_layout()
plt.savefig('target_distribution_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print('Feature engineering')


def sanity_check_targets(df, cat_col='investment_cat', ratio_col='saving_ratio'):
    """Validasi target sebelum training. Jika imbalanced ekstrem, beri warning."""
    print('\n' + '='*55)
    print('  🔍 SANITY CHECK: Distribusi Target')
    print('='*55)

    # Cek ratio
    print(f'\nSaving Ratio Stats:')
    print(f'  Mean  : {df[ratio_col].mean():.4f}')
    print(f'  Std   : {df[ratio_col].std():.4f}')
    print(f'  Min   : {df[ratio_col].min():.4f}')
    print(f'  Max   : {df[ratio_col].max():.4f}')

    if df[ratio_col].std() < 0.05:
        print('      WARNING: Std sangat rendah → target hampir seragam!')
        print('      Model akan belajar prediksi mean saja (useless).')
    else:
        print('  Variasi cukup untuk training.')

    # Cek imbalance klasifikasi
    vc = df[cat_col].value_counts(normalize=True)
    max_pct = vc.max()
    print(f'\nInvestment Category Balance:')
    for k, pct in vc.items():
        bar = '█' * int(pct * 40)
        print(f'  Cat {k}: {pct*100:5.1f}% {bar}')

    if max_pct > 0.80:
        print('  ⚠️  WARNING: Imbalance ekstrem! Gunakan class_weight atau oversampling.')
    elif max_pct > 0.60:
        print('  ⚠️  Cukup imbalanced. Pertimbangkan class_weight.')
    else:
        print('  ✅ Balance distribusi ok.')

    print('='*55)

sanity_check_targets(df)

# Feature set
FEATURES = [
    'income_log',        # log-transform income (tidak skewed)
    'expense_ratio',     # rasio bulanan yang BENAR
    'lifestyle_burden',  # beban gaya hidup relatif
    'debt_burden',       # beban cicilan (normalized)
    'credit_norm',       # credit score normalized
    'risk_status',       # preferensi risiko user
    'family_size',       # tanggungan
    'education',         # pendidikan
    'age',               # usia
    'budget_encoded',    # status anggaran (Good=2/Average=1/Bad=0)
]

X  = df[FEATURES].values.astype(np.float32)
y_ratio = df['saving_ratio'].values.astype(np.float32)    # Target regresi
y_cat   = df['investment_cat'].values.astype(np.int32)     # Target klasifikasi


from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(y_cat), y=y_cat)
class_weights = {i: cw[i] for i in range(len(cw))}
print('Class Weights (untuk imbalance handling):',
      {inv_names[k]: round(v, 2) for k, v in class_weights.items()})


X_tmp, X_test, yr_tmp, yr_test, yc_tmp, yc_test = train_test_split(
    X, y_ratio, y_cat, test_size=0.15, random_state=SEED, stratify=y_cat)
X_train, X_val, yr_train, yr_val, yc_train, yc_val = train_test_split(
    X_tmp, yr_tmp, yc_tmp, test_size=0.176, random_state=SEED, stratify=yc_tmp)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

print(f'\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')
print(f'Features: {FEATURES}')
print('\n Preprocessing v2 selesai!')


# FinancialAttentionLayer


class FinancialAttentionLayer(keras.layers.Layer):
    """
    Custom Layer dengan Feature-wise Attention.
    Belajar bobot kepentingan setiap fitur keuangan secara dinamis.
    Versi v2 menggunakan temperature scaling untuk stabilitas training.
    """
    def __init__(self, units=64, temperature=1.0, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units        = units
        self.temperature  = temperature
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        n_feat = input_shape[-1]
        # Attention query vector
        self.attn_w = self.add_weight(
            name='attn_w', shape=(n_feat, n_feat),
            initializer='glorot_uniform', trainable=True)
        self.attn_b = self.add_weight(
            name='attn_b', shape=(n_feat,),
            initializer='zeros', trainable=True)
        # Projection
        self.proj_w = self.add_weight(
            name='proj_w', shape=(n_feat, self.units),
            initializer='glorot_uniform', trainable=True)
        self.proj_b = self.add_weight(
            name='proj_b', shape=(self.units,),
            initializer='zeros', trainable=True)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=False):

        scores = (tf.matmul(inputs, self.attn_w) + self.attn_b) / self.temperature
        weights = tf.nn.softmax(scores, axis=-1)  # (batch, n_feat)

        attended = inputs * weights

        out = tf.nn.swish(tf.matmul(attended, self.proj_w) + self.proj_b)
        return self.dropout(out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'temperature': self.temperature,
                       'dropout_rate': self.dropout_rate})
        return config



# CUSTOM LOSS HybridFinancialLoss
# Kombinasi MSE + Huber untuk robustness terhadap outlier income


class HybridFinancialLoss(keras.losses.Loss):
    """
    Custom Loss Function:
    - MSE untuk saving_ratio < 0.35 (high-risk group, penalti besar)
    - Huber loss untuk saving_ratio >= 0.35 (robust terhadap outlier)
    Mengatasi masalah sigmoid saturation dari v1.
    """
    def __init__(self, risk_threshold=0.35, risk_weight=2.5, delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.risk_threshold = risk_threshold
        self.risk_weight    = risk_weight
        self.delta          = delta

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        err    = y_true - y_pred


        huber  = tf.where(
            tf.abs(err) <= self.delta,
            0.5 * tf.square(err),
            self.delta * (tf.abs(err) - 0.5 * self.delta)
        )

        weights = tf.where(
            y_true < self.risk_threshold,
            tf.ones_like(y_true) * self.risk_weight,
            tf.ones_like(y_true)
        )
        return tf.reduce_mean(weights * huber)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'risk_threshold': self.risk_threshold,
                    'risk_weight': self.risk_weight, 'delta': self.delta})
        return cfg



# CUSTOM CALLBACK v2: FinancialHealthMonitor

class FinancialHealthMonitor(keras.callbacks.Callback):
    """
    Callback yang memantau:
    1. Training progress setiap N epoch
    2. Distribusi prediksi pada validation set (deteksi saturation)
    3. Alert jika output model terlalu seragam (bug v1)
    """
    def __init__(self, X_val, report_every=5):
        super().__init__()
        self.X_val        = X_val
        self.report_every = report_every
        self.history      = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))

        if (epoch + 1) % self.report_every == 0:

            preds = self.model.predict(self.X_val, verbose=0)
            ratios = preds['ratio_output'].flatten()
            cats   = np.argmax(preds['cat_output'], axis=1)

            ratio_std = ratios.std()
            cat_vc    = np.bincount(cats, minlength=3) / len(cats)

            health = ' Sehat'
            if ratio_std < 0.02:
                health = ' SATURASI! Output seragam — cek learning rate & data'
            elif ratio_std < 0.05:
                health = ' Peringatan: variasi prediksi rendah'

            print(f'\n[Epoch {epoch+1:3d}] Loss={logs["loss"]:.4f} ValLoss={logs["val_loss"]:.4f}')
            print(f'  Ratio → mean={ratios.mean():.3f} std={ratios.std():.3f}  {health}')
            print(f'  Cat   → Reksadana={cat_vc[0]:.1%} Obligasi={cat_vc[1]:.1%} Saham={cat_vc[2]:.1%}')

    def on_train_begin(self, logs=None):
        print('\n✨ Training dimulai — v2 dengan saturation detection\n')

print('✅ Semua custom komponen v2 siap!')

def build_model_v2(n_features=10):

    inp = keras.Input(shape=(n_features,), name='financial_features')

    #  Attention Embedding
    x = FinancialAttentionLayer(
        units=64, temperature=2.0, dropout_rate=0.2,
        name='financial_attention')(inp)

    h1 = layers.Dense(128, activation='swish', name='bb_1')(x)
    h1 = layers.BatchNormalization(name='bn_1')(h1)
    h1 = layers.Dropout(0.3, name='drop_1')(h1)

    h2 = layers.Dense(64, activation='swish', name='bb_2')(h1)
    h2 = layers.BatchNormalization(name='bn_2')(h2)
    h2 = layers.Dropout(0.2, name='drop_2')(h2)

    x_res  = layers.Dense(64, name='residual_proj')(x)
    shared = layers.Add(name='residual_add')([h2, x_res])
    shared = layers.Activation('swish', name='shared_act')(shared)

    # Head 1: Saving Ratio Regression
    r = layers.Dense(32, activation='swish', name='ratio_h1')(shared)
    r = layers.Dense(16, activation='swish', name='ratio_h2')(r)

    ratio_out = layers.Dense(1, activation='linear', name='ratio_output')(r)

    # Head 2: Investment Category Classification
    c = layers.Dense(32, activation='swish', name='cat_h1')(shared)
    c = layers.Dense(16, activation='swish', name='cat_h2')(c)
    cat_out = layers.Dense(3, activation='softmax', name='cat_output')(c)

    model = Model(
        inputs=inp,
        outputs={'ratio_output': ratio_out, 'cat_output': cat_out},
        name='SmartBudgeting_v2'
    )
    return model


model = build_model_v2(n_features=len(FEATURES))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss={
        'ratio_output': HybridFinancialLoss(name='hybrid_loss'),
        'cat_output'  : keras.losses.SparseCategoricalCrossentropy()
    },
    loss_weights={'ratio_output': 0.2, 'cat_output': 1.8},
    metrics={
        'ratio_output': [keras.metrics.MeanAbsoluteError(name='mae')],
        'cat_output'  : [keras.metrics.SparseCategoricalAccuracy(name='acc')]
    }
)

model.summary()
print('\n✅ Model v2 compiled!')

y_train_dict = {'ratio_output': yr_train, 'cat_output': yc_train}
y_val_dict   = {'ratio_output': yr_val,   'cat_output': yc_val}

# Sample weights: atasi imbalance pada klasifikasi
sample_weights = np.array([class_weights[c] for c in yc_train])

monitor_cb = FinancialHealthMonitor(X_val=X_val_s, report_every=5)

callbacks = [
    monitor_cb,
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-3, verbose=1),
    keras.callbacks.ModelCheckpoint(
        'best_robo_advisor_v2.keras', monitor='val_loss',
        save_best_only=True, verbose=0)
]

print("=== CEK DISTRIBUSI TARGET ===")
print(f"Min Saving Ratio: {df['saving_ratio'].min()}")
print(f"Max Saving Ratio: {df['saving_ratio'].max()}")
print(f"Mean Saving Ratio: {df['saving_ratio'].mean()}")

history = model.fit(
    X_train_s, y_train_dict,
    validation_data=(X_val_s, y_val_dict),
    sample_weight=sample_weights,
    epochs=150,
    batch_size=256,
    callbacks=callbacks,
    verbose=0
)

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(' Training History v2', fontsize=13, fontweight='bold')

axes[0].plot(history.history['loss'], label='Train'); axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

mae_k = [k for k in history.history if 'mae' in k and 'val' not in k]
val_mae_k = [k for k in history.history if 'mae' in k and 'val' in k]
if mae_k:
    axes[1].plot(history.history[mae_k[0]], label='Train MAE')
    axes[1].plot(history.history[val_mae_k[0]], label='Val MAE')
axes[1].set_title('Saving Ratio MAE'); axes[1].legend(); axes[1].grid(alpha=0.3)

acc_k = [k for k in history.history if 'acc' in k and 'val' not in k]
val_acc_k = [k for k in history.history if 'acc' in k and 'val' in k]
if acc_k:
    axes[2].plot(history.history[acc_k[0]], label='Train Acc')
    axes[2].plot(history.history[val_acc_k[0]], label='Val Acc')
axes[2].set_title('Category Accuracy'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_v2.png', dpi=150, bbox_inches='tight')
plt.show()

# Test set
preds    = model.predict(X_test_s, verbose=0)
p_ratios = preds['ratio_output'].flatten()
p_cats   = np.argmax(preds['cat_output'], axis=1)

mae_val  = np.mean(np.abs(p_ratios - yr_test))
print(f'\n TEST SET RESULTS:')
print(f'  Saving Ratio MAE : {mae_val:.4f} ({mae_val*100:.2f}%)')
print(f'  Ratio Pred Std   : {p_ratios.std():.4f}  ← harus > 0.05, bukan 0.001')
print(f'  Ratio Pred Range : {p_ratios.min():.3f} – {p_ratios.max():.3f}  ← harus bervariasi')
print()
print(classification_report(yc_test, p_cats,
      target_names=['Reksadana', 'Obligasi', 'Saham']))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(yc_test, p_cats)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Reksadana','Obligasi','Saham'],
            yticklabels=['Reksadana','Obligasi','Saham'], ax=ax)
ax.set_title('Confusion Matrix v2'); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_v2.png', dpi=150, bbox_inches='tight')
plt.show()


model.save('robo_advisor_v2.keras')
print('robo_advisor_v2.keras')


model.export('robo_advisor_v2_savedmodel')
print('robo_advisor_v2_savedmodel/')

# Preprocessing artifacts
with open('scaler_v2.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('features_v2.pkl', 'wb') as f: pickle.dump(FEATURES, f)
print(' scaler_v2.pkl, features_v2.pkl')

loaded = keras.models.load_model(
    'robo_advisor_v2.keras',
    custom_objects={'FinancialAttentionLayer': FinancialAttentionLayer,
                    'HybridFinancialLoss': HybridFinancialLoss}
)
vp = loaded.predict(X_test_s[:3], verbose=0)
print(f'\n Load OK | Ratios: {vp["ratio_output"].flatten().round(3)}')
print(f'   Cats  : {np.argmax(vp["cat_output"], axis=1)}')
