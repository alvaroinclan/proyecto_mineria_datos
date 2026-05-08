"""

tratamiento_datos.py  -  Tratamiento de Valores Faltantes
Dataset: Smartphone Addiction Prediction Data  (data base.csv)
Trabajo Final - Minería de Datos - Grado en Matemáticas

Nos centramos en la única variable con valores faltantes: addiction_level.
Diagnosticamos el mecanismo de pérdida (MCAR/MAR/MNAR), justificamos
estadísticamente la decisión adoptada y generamos gráficos de apoyo.

"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data base.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")
os.makedirs(IMG_DIR, exist_ok=True)

# Carga de datos
df = pd.read_csv(DATA_PATH)

# Definimos máscara para datos faltantes en variable addictionlevel
mask_nan = df["addiction_level"].isnull()
n_nan = mask_nan.sum()

## Usamos == como separador entre secciones para organizar el código que si no me lío


# ============================================================================
# 1. LOCALIZACIÓN DE LOS FALTANTES
# ============================================================================
print("\n")
print("1. LOCALIZACIÓN DE LOS FALTANTES")
## Realmente ya lo sabíamos del EDA pero bueno para asegurar y que quede bonito

print(
    f"\n  Registros con NaN en addiction_level: {n_nan:,} de {len(df):,} "
    f"({n_nan / len(df) * 100:.2f}%)"
)
print("  Resto de variables: 0 valores faltantes.")

# Qué valor tiene addicted_label en las filas con NaN
print("\n  addicted_label en filas con NaN:")
for val, cnt in df.loc[mask_nan, "addicted_label"].value_counts().items():
    print(f"    addicted_label = {val}: {cnt:,} ({cnt / n_nan * 100:.1f}%)")

print("\n  addicted_label en filas SIN NaN:")
for val, cnt in df.loc[~mask_nan, "addicted_label"].value_counts().items():
    print(f"    addicted_label = {val}: {cnt:,} ({cnt / (~mask_nan).sum() * 100:.1f}%)")


# ============================================================================
# 2. DIAGNÓSTICO DEL MECANISMO DE PÉRDIDA (por qué perdemos los datos básicamente)
# ============================================================================
print("\n")
print("2. DIAGNÓSTICO DEL MECANISMO DE PÉRDIDA (MCAR / MAR / MNAR)")


# Test Chi2 de independencia: ver si ser NaN depende de addicted_label
contingency = pd.crosstab(mask_nan, df["addicted_label"])
chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
print("\n  Test Chi2 (NaN vs addicted_label):")
print(f"    chi2 = {chi2:.4f},  gl = {dof},  p = {p_chi2:.4e}")
if p_chi2 < 0.05:
    print("Se rechaza H0: los NaN dependen del target - mecanismo MAR.")
else:
    print("No se rechaza H0: compatible con MCAR.")

# Test t de medias en las numéricas principales
## El objetivo es determinar si hay sesgo en la ausencia de datos o es aleatoria
num_cols = [
    "daily_screen_time_hours",
    "social_media_hours",
    "sleep_hours",
    "notifications_per_day",
]
print("\n  Test t (Welch) - medias de filas con NaN vs sin NaN:")
print(
    f"  {'Variable':30s} {'Media NaN':>10s} {'Media OK':>10s} "
    f"{'p-valor':>12s} {'Signif':>7s}"
)
print("  " + "─" * 72)
for col in num_cols:
    g_nan = df.loc[mask_nan, col].dropna()
    g_ok = df.loc[~mask_nan, col].dropna()  # ~ operador logico NOT para bool
    _, p = stats.ttest_ind(g_nan, g_ok, equal_var=False)
    sig = "Sí *" if p < 0.05 else "No"
    print(f"  {col:30s} {g_nan.mean():10.3f} {g_ok.mean():10.3f} {p:12.4e} {sig:>7s}")

# Gráfico: distribuciones NaN vs OK
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(
        df.loc[~mask_nan, col],
        bins=30,
        alpha=0.6,  # ~ operador logico NOT para bool
        color="#3498db",
        label="OK",
        density=True,
    )
    axes[i].hist(
        df.loc[mask_nan, col],
        bins=30,
        alpha=0.6,
        color="#e74c3c",
        label="NaN",
        density=True,
    )
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)
plt.suptitle(
    "Distribuciones: filas con NaN vs sin NaN en addiction_level", fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "10_distribucion_nan_vs_ok.png"))
plt.close()


# ============================================================================
# 3. RELACIÓN DETERMINISTA CON EL TARGET
# ============================================================================
print("\n")
print("3. RELACIÓN DETERMINISTA: addiction_level y addicted_label")


ct = pd.crosstab(
    df["addiction_level"].fillna("(NaN)"), df["addicted_label"], margins=True
)
print("\n  Tabla cruzada:")
print(ct.to_string())

# Verificar determinismo
df_ok = df.dropna(subset=["addiction_level"])
mild_en_1 = (
    (df_ok["addiction_level"] == "Mild") & (df_ok["addicted_label"] == 1)
).sum()
modsev_en_0 = (
    (df_ok["addiction_level"].isin(["Moderate", "Severe"]))
    & (df_ok["addicted_label"] == 0)
).sum()
print("\n  Excepciones al determinismo:")
print(f"    Mild con label=1:            {mild_en_1}")
print(f"    Moderate/Severe con label=0: {modsev_en_0}")

determinista = mild_en_1 == 0 and modsev_en_0 == 0
if determinista:
    print("  Relación 100% determinista confirmada.")

# Gráfico: tabla cruzada visual
fig, ax = plt.subplots(figsize=(8, 5))
ct_plot = pd.crosstab(df["addiction_level"].fillna("(NaN)"), df["addicted_label"])
ct_plot.plot(kind="bar", stacked=True, ax=ax, color=["#3498db", "#e74c3c"], alpha=0.85)
ax.set_title("addiction_level - addicted_label")
ax.set_xlabel("addiction_level")
ax.set_ylabel("Frecuencia")
ax.legend(title="addicted_label")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "11_addiction_level_vs_target.png"))
plt.close()
