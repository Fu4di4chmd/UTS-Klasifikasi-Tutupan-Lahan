
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. DIREKTORI
# =====================================================

BASE = Path(__file__).resolve().parent.parent
REPORT = BASE / "outputs" / "reports"
CHARTS = BASE / "outputs" / "charts"

CHARTS.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "build up",
    "farm land",
    "water body",
    "forest",
    "mangrove"
]

sns.set_theme(style="whitegrid")

# =====================================================
# 2. BACA HASIL MODEL
# =====================================================

with open(REPORT / "metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

cm = pd.read_csv(
    REPORT / "confusion_matrix.csv",
    index_col=0
)

importance = pd.read_csv(
    REPORT / "feature_importance.csv"
)

report = pd.read_csv(
    REPORT / "classification_report.csv",
    index_col=0
)

training = pd.read_csv(REPORT / "training_balanced.csv")
testing = pd.read_csv(REPORT / "testing.csv")

# =====================================================
# 3. RINGKASAN METRIK
# =====================================================

nama_metrik = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "Weighted F1"
]

nilai = [
    metrics["accuracy"],
    metrics["balanced_accuracy"],
    metrics["macro_f1"],
    metrics["weighted_f1"]
]

plt.figure(figsize=(10, 5))

ax = sns.barplot(
    x=nama_metrik,
    y=nilai,
    color="#2979B9"
)

for i, v in enumerate(nilai):
    ax.text(
        i,
        v + 0.015,
        f"{v * 100:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.ylim(0, 1.1)
plt.ylabel("Nilai")
plt.title("Evaluasi Model Random Forest")

plt.tight_layout()

plt.savefig(
    CHARTS / "evaluasi_model.png",
    dpi=200
)

plt.close()

# =====================================================
# 4. CONFUSION MATRIX
# =====================================================

cm = cm.loc[CLASSES, CLASSES]

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Prediksi Model")
plt.ylabel("Label Sebenarnya")

plt.tight_layout()

plt.savefig(
    CHARTS / "confusion_matrix.png",
    dpi=200
)

plt.close()

# =====================================================
# 5. FEATURE IMPORTANCE
# =====================================================

importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    importance["Feature"],
    importance["Importance"],
    color="#2979B9"
)

plt.xlabel("Feature Importance")
plt.title("Kontribusi 14 Fitur Random Forest")

plt.tight_layout()

plt.savefig(
    CHARTS / "feature_importance.png",
    dpi=200
)

plt.close()

# =====================================================
# 6. DISTRIBUSI SAMPEL TRAINING DAN TESTING
# =====================================================

train_counts = (
    training["label"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)

test_counts = (
    testing["label"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)

distribusi = pd.DataFrame({
    "Training": train_counts,
    "Testing": test_counts
})

distribusi.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Distribusi Sampel Training dan Testing")
plt.ylabel("Jumlah Piksel")
plt.xlabel("Kelas Tutupan Lahan")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    CHARTS / "distribusi_sampel.png",
    dpi=200
)

plt.close()

# =====================================================
# 7. PERFORMA TIAP KELAS
# =====================================================

per_class = report.loc[
    CLASSES,
    ["precision", "recall", "f1-score"]
]

per_class.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Precision, Recall, dan F1-score Setiap Kelas")
plt.xlabel("Kelas Tutupan Lahan")
plt.ylabel("Nilai")
plt.ylim(0, 1.1)
plt.xticks(rotation=25)
plt.legend(title="Metrik")

plt.tight_layout()

plt.savefig(
    CHARTS / "performa_per_kelas.png",
    dpi=200
)

plt.close()

# =====================================================
# 8. SELESAI
# =====================================================

print("=" * 55)
print("✅ EVALUATION BERHASIL")
print("=" * 55)

print("Accuracy:", f"{metrics['accuracy'] * 100:.2f}%")
print("Balanced Accuracy:", f"{metrics['balanced_accuracy'] * 100:.2f}%")
print("Macro F1:", f"{metrics['macro_f1'] * 100:.2f}%")
print("Weighted F1:", f"{metrics['weighted_f1'] * 100:.2f}%")

print("\nGrafik tersimpan di:", CHARTS)

for file in CHARTS.glob("*.png"):
    print("✅", file.name)
