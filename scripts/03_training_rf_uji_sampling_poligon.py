# ============================================================
# EKSPERIMEN TRAINING RANDOM FOREST - FITUR MANGROVE BARU
# Menyimpan model dan laporan terpisah (tidak menimpa baseline).
# Jalankan: py -3.9 scripts/03_training_rf_uji_sampling_poligon.py
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE / 'outputs' / 'reports'
# Menggunakan folder yang sama dengan output preprocessing
EXPERIMENT_DIR = REPORT_DIR / 'eksperimen_mangrove_17fitur'
MODEL_DIR = BASE / 'outputs' / 'models'
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = EXPERIMENT_DIR / 'training_17fitur.csv'
TEST_PATH = EXPERIMENT_DIR / 'testing_17fitur.csv'
MODEL_PATH = MODEL_DIR / 'random_forest_model_mangrove_17fitur.joblib'

BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
# --- PERBAIKAN: Menambahkan MANGROVE_8A_11 dan VEG_8A_4 ke dalam model ---
INDEKS = ['NDVI', 'NDMI', 'MNDWI', 'NDBI', 'NDWI', 'MVI', 'REMI', 'MANGROVE_8A_11', 'VEG_8A_4']
FEATURES = BANDS + INDEKS
CLASSES = ['build up', 'farm land', 'water body', 'forest', 'mangrove']
MAX_SAMPLES = 2000
SEED = 42

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print('=' * 60)
print('EKSPERIMEN RANDOM FOREST - FITUR MANGROVE BARU')
print('=' * 60)
print('Training awal:', train_df.shape)
print('Testing:', test_df.shape)

required_columns = FEATURES + ['polygon_id', 'label']
for column in required_columns:
    if column not in train_df.columns or column not in test_df.columns:
        raise ValueError(f'Kolom wajib tidak ditemukan: {column}')
train_ids = set(train_df['polygon_id'].unique())
test_ids = set(test_df['polygon_id'].unique())
assert train_ids.isdisjoint(test_ids), 'Kebocoran ID poligon train/test!'
assert np.isfinite(train_df[FEATURES].to_numpy()).all(), 'Training memiliki NaN/Inf'
assert np.isfinite(test_df[FEATURES].to_numpy()).all(), 'Testing memiliki NaN/Inf'

balanced_parts = []
for kelas in CLASSES:
    bagian = train_df.loc[train_df['label'] == kelas]
    if bagian.empty:
        raise ValueError(f'Kelas tidak memiliki training: {kelas}')
    jumlah = min(len(bagian), MAX_SAMPLES)

    if kelas == 'forest':
        # Water-filling: setiap poligon mendapat giliran satu piksel sampai
        # mencapai kapasitas masing-masing atau kuota Forest 2000 terpenuhi.
        kelompok = {pid: frame for pid, frame in bagian.groupby('polygon_id', sort=True)}
        rng = np.random.RandomState(SEED)
        ids = list(kelompok)
        rng.shuffle(ids)
        kapasitas = {pid: len(kelompok[pid]) for pid in ids}
        alokasi = {pid: 0 for pid in ids}
        sisa = jumlah
        while sisa:
            tambah = 0
            for pid in ids:
                if sisa == 0:
                    break
                if alokasi[pid] < kapasitas[pid]:
                    alokasi[pid] += 1
                    sisa -= 1
                    tambah += 1
            if tambah == 0:
                raise RuntimeError('Tidak dapat mengalokasikan seluruh kuota Forest')
        sampel = pd.concat([
            kelompok[pid].sample(n=alokasi[pid], random_state=SEED)
            for pid in ids if alokasi[pid] > 0
        ], ignore_index=True)
        print(f'Forest: {len(sampel)} piksel dari {sampel.polygon_id.nunique()} poligon')
        print('Distribusi sampel Forest per poligon:')
        print(sampel.groupby('polygon_id').size().sort_values(ascending=False).to_string())
    else:
        sampel = bagian.sample(n=jumlah, random_state=SEED)
    balanced_parts.append(sampel)

balanced_df = pd.concat(balanced_parts, ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
print('\nDistribusi training setelah sampling:')
print(balanced_df['label'].value_counts().to_string())
print('Total training:', len(balanced_df), '| Testing tetap:', len(test_df))
balanced_df.to_csv(EXPERIMENT_DIR / 'training_balanced.csv', index=False)

X_train = balanced_df[FEATURES].to_numpy(dtype=np.float32)
y_train = balanced_df['label'].to_numpy()
X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
y_test = test_df['label'].to_numpy()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    min_samples_leaf=1,
    random_state=SEED,
    class_weight='balanced',
    n_jobs=-1,
)
print('\nMelatih model eksperimen...')
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, labels=CLASSES, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, labels=CLASSES, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

print('\n' + '=' * 60)
print('HASIL EVALUASI MODEL EKSPERIMEN')
print('=' * 60)
print(f'Accuracy         : {accuracy * 100:.2f}%')
print(f'Balanced Accuracy: {balanced_accuracy * 100:.2f}%')
print(f'Macro F1-Score   : {macro_f1 * 100:.2f}%')
print(f'Weighted F1     : {weighted_f1 * 100:.2f}%')
print('\nCLASSIFICATION REPORT')
print(classification_report(y_test, y_pred, labels=CLASSES,
                            target_names=CLASSES, digits=4, zero_division=0))
print('\nCONFUSION MATRIX')
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

importance_df = pd.DataFrame({
    'Feature': FEATURES, 'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print('\nFEATURE IMPORTANCE')
print(importance_df.to_string(index=False))

report_df = pd.DataFrame(classification_report(
    y_test, y_pred, labels=CLASSES, target_names=CLASSES,
    output_dict=True, zero_division=0,
)).transpose()
report_df.to_csv(EXPERIMENT_DIR / 'classification_report.csv')
importance_df.to_csv(EXPERIMENT_DIR / 'feature_importance.csv', index=False)
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
    EXPERIMENT_DIR / 'confusion_matrix.csv'
)
hasil_test = test_df[['polygon_id', 'label']].copy()
hasil_test['prediksi_rf'] = y_pred
hasil_test.to_csv(EXPERIMENT_DIR / 'hasil_prediksi_testing.csv', index=False)

metrics = {
    'accuracy': float(accuracy),
    'balanced_accuracy': float(balanced_accuracy),
    'macro_f1': float(macro_f1),
    'weighted_f1': float(weighted_f1),
    'jumlah_training': int(len(balanced_df)),
    'jumlah_testing': int(len(test_df)),
    'jumlah_fitur': len(FEATURES),
    'fitur': FEATURES,
    'precision_mangrove': float(report_df.loc['mangrove','precision']),
    'recall_mangrove': float(report_df.loc['mangrove','recall']),
    'f1_mangrove': float(report_df.loc['mangrove','f1-score']),
    'jumlah_poligon_training': len(train_ids),
    'jumlah_poligon_testing': len(test_ids),
    'metode': 'forest sampling merata per polygon_id; kelas lain random sample',
}
with (EXPERIMENT_DIR / 'metrics.json').open('w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=4)

joblib.dump({'model': rf_model, 'features': FEATURES, 'classes': CLASSES}, MODEL_PATH)
print('\n' + '=' * 60)
print('EKSPERIMEN SELESAI')
print('Model eksperimen:', MODEL_PATH)
print('Laporan eksperimen:', EXPERIMENT_DIR)
print('Data testing tidak diubah; tidak ada kebocoran poligon.')