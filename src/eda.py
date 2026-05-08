"""

eda.py  -  Análisis Exploratorio de Datos (EDA)
Dataset: Smartphone Addiction Prediction Data  (data base.csv)
Trabajo Final - Minería de Datos - Grado en Matemáticas

Generamos gráficos en  docs/img/  y un output por consola
que se usará para redactar el informe en  docs/index.md.

"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
# Definimos parámetros de los gráficos
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


# Clasificación de variables para luego utilizar en el código
ID_COLS = ["transaction_id", "user_id"]

NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

CAT_ORDINAL_COLS = ["stress_level", "addiction_level"]
CAT_NOMINAL_COLS = ["gender", "academic_work_impact"]
TARGET_COL = "addicted_label"

ALL_CAT = CAT_ORDINAL_COLS + CAT_NOMINAL_COLS + [TARGET_COL]

## Usamos == como separador entre secciones para organizar el código que si no me lío

# ============================================================================
# 1. VISIÓN GENERAL DEL DATASET
# ============================================================================

print("1. VISIÓN GENERAL")

print(f"  Filas:     {df.shape[0]:,}")
print(f"  Columnas:  {df.shape[1]}")
print("\n  Tipos de datos:")
for dtype, count in df.dtypes.value_counts().items():
    print(f"    {dtype}: {count}")

print("\n  Primeras 5 filas:")
print(df.head())

print("\n  Información de columnas:")
for col in df.columns:
    print(
        f"    {col:30s}  dtype={str(df[col].dtype):8s}  "
        f"no-null={df[col].notna().sum():6,}  "
        f"unique={df[col].nunique():6,}"
    )

# ============================================================================
# 2. ANÁLISIS DE VALORES FALTANTES
# ============================================================================
print("\n")
print("2. VALORES FALTANTES (NaN)")


null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_df = pd.DataFrame({"Faltantes": null_counts, "Porcentaje (%)": null_pct})
null_df = null_df.sort_values("Porcentaje (%)", ascending=False)

if null_df["Faltantes"].sum() == 0:
    print("  No se detectaron valores faltantes (NaN) en el dataset.")
else:
    null_df_show = null_df[null_df["Faltantes"] > 0]
    print(null_df_show.to_string())

print(
    f"\n  Total de celdas faltantes: {null_counts.sum():,} "
    f"({null_counts.sum() / df.size * 100:.2f}% del total)"
)

# Gráfico de mapa de faltantes
fig, ax = plt.subplots(figsize=(12, 5))
null_matrix = df.isnull().astype(int)
sample_idx = np.sort(np.random.choice(len(df), size=min(500, len(df)), replace=False))
sns.heatmap(
    null_matrix.iloc[sample_idx].T,
    cbar=False,
    cmap="YlOrRd",
    yticklabels=df.columns,
    xticklabels=False,
    ax=ax,
)
ax.set_title("Mapa de valores faltantes (muestra de 500 filas)")
ax.set_xlabel("Observaciones")
ax.set_ylabel("Variables")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "01_mapa_faltantes.png"))
plt.close()

# ============================================================================
# 3. ESTADÍSTICOS DESCRIPTIVOS - VARIABLES NUMÉRICAS
# ============================================================================
print("\n")
print("3. ESTADÍSTICOS DESCRIPTIVOS - VARIABLES NUMÉRICAS")


desc = df[NUM_COLS].describe().T  # Transponemos para que se vea bien la tabla
desc["skew"] = df[NUM_COLS].skew()
desc["kurtosis"] = df[NUM_COLS].kurtosis()
print(desc.round(3).to_string())

# ============================================================================
# 4. DISTRIBUCIONES – VARIABLES NUMÉRICAS
# ============================================================================
print("\n")
print("4. DISTRIBUCIONES - VARIABLES NUMÉRICAS")


n_num = len(NUM_COLS)
fig, axes = plt.subplots(n_num, 2, figsize=(14, 4 * n_num))
for i, col in enumerate(NUM_COLS):
    s = df[col].dropna()
    # Histograma
    axes[i, 0].hist(s, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[i, 0].axvline(
        s.mean(), color="#e74c3c", ls="--", label=f"Media={s.mean():.2f}"
    )
    axes[i, 0].axvline(
        s.median(), color="#2ecc71", ls="-", label=f"Mediana={s.median():.2f}"
    )
    axes[i, 0].legend(fontsize=8)
    axes[i, 0].set_title(f"Histograma: {col}")
    axes[i, 0].set_ylabel("Frecuencia")
    # Boxplot
    axes[i, 1].boxplot(
        s, vert=False, patch_artist=True, boxprops=dict(facecolor="#3498db", alpha=0.6)
    )
    axes[i, 1].set_title(f"Boxplot: {col}")

plt.suptitle("Distribuciones - Variables Numéricas", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "02_distribuciones_numericas.png"))
plt.close()


# Test de normalidad (Shapiro-Wilk sobre muestra)para ver si siguen una distribución normal.
print("\n  Test de normalidad (Shapiro-Wilk, muestra n≤5000):")
print(f"  {'Variable':30s} {'Estadístico':>12s} {'p-valor':>12s} {'Normal?':>10s}")

for col in NUM_COLS:
    s = df[col].dropna()
    if len(s) < 10:
        continue
    sample = s.sample(min(5000, len(s)), random_state=42)
    stat, pval = stats.shapiro(sample)
    is_normal = "Sí" if pval > 0.05 else "No"
    print(f"  {col:30s} {stat:12.6f} {pval:12.2e} {is_normal:>10s}")

# ============================================================================
# 5. DISTRIBUCIONES - VARIABLES CATEGÓRICAS
# ============================================================================
print("\n")
print("5. DISTRIBUCIONES - VARIABLES CATEGÓRICAS")


for col in ALL_CAT:
    vc = df[col].value_counts()
    n_valid = df[col].notna().sum()
    print(f"\n  {col} ({vc.shape[0]} categorías, {df[col].isna().sum()} NaN):")
    for cat, cnt in vc.items():
        pct = cnt / n_valid * 100
        print(f"    {str(cat):25s}  {cnt:6,}  ({pct:5.1f}%)")

# Gráficos de barras
n_cat = len(ALL_CAT)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
palette = sns.color_palette("Set2", 20)
for i, col in enumerate(ALL_CAT):
    if i >= len(axes):
        break
    vc = df[col].value_counts()
    axes[i].barh(vc.index[::-1], vc.values[::-1], color=palette[: len(vc)])
    axes[i].set_title(f"Distribución: {col}")
    axes[i].set_xlabel("Frecuencia")

# Ocultar ejes sobrantes
for j in range(n_cat, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Distribuciones - Variables Categóricas y Objetivo", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "03_distribuciones_categoricas.png"))
plt.close()


# ============================================================================
# 6. ANÁLISIS DE CORRELACIONES
# ============================================================================
print("\n")
print("6. ANÁLISIS DE CORRELACIONES")


df_num = df[NUM_COLS + [TARGET_COL]].copy()

# 6a. Pearson
corr_pearson = df_num.corr(method="pearson")
print("\n  Matriz de correlación de Pearson:")
print(corr_pearson.round(3).to_string())

# 6b. Spearman
corr_spearman = df_num.corr(method="spearman")
print("\n  Matriz de correlación de Spearman:")
print(corr_spearman.round(3).to_string())

# Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

sns.heatmap(
    corr_pearson,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=axes[0],
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
axes[0].set_title("Correlación de Pearson")

sns.heatmap(
    corr_spearman,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=axes[1],
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
axes[1].set_title("Correlación de Spearman")

plt.suptitle("Matrices de Correlación", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "04_correlaciones.png"))
plt.close()


# Top correlaciones
print("\n  Top correlaciones (|r| > 0.3, Pearson):")
corr_pairs = []
cols = corr_pearson.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = corr_pearson.iloc[i, j]
        if abs(r) > 0.3:
            corr_pairs.append((cols[i], cols[j], r))

corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
if corr_pairs:
    for v1, v2, r in corr_pairs:
        print(f"    {v1} <-> {v2}: r = {r:.4f}")
else:
    print("    No se encontraron correlaciones |r| > 0.3")

# Correlaciones con la variable objetivo
print(f"\n  Correlaciones con {TARGET_COL} (Pearson):")
target_corr = (
    corr_pearson[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
)
for var, r in target_corr.items():
    marker = (
        "***" if abs(r) > 0.5 else "**" if abs(r) > 0.3 else "*" if abs(r) > 0.1 else ""
    )
    print(f"    {var:30s}: r = {r:+.4f}  {marker}")

# 6c. Scatter plots de las relaciones más relevantes con la variable objetivo
top_corr_vars = target_corr.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, var in enumerate(top_corr_vars):
    sx = df[var].dropna()
    sy = df[TARGET_COL].dropna()
    common = sx.index.intersection(sy.index)
    sample_idx = np.random.choice(common, size=min(3000, len(common)), replace=False)
    axes[i].scatter(
        df.loc[sample_idx, var],
        df.loc[sample_idx, TARGET_COL],
        alpha=0.2,
        s=8,
        color="#3498db",
    )
    r_val = df.loc[common, [var, TARGET_COL]].corr().iloc[0, 1]
    axes[i].set_xlabel(var)
    axes[i].set_ylabel(TARGET_COL)
    axes[i].set_title(f"{var} vs {TARGET_COL}  (r={r_val:.3f})")

plt.suptitle(
    "Diagramas de Dispersión - Top correlaciones con addicted_label", fontsize=14
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "05_scatter_target.png"))
plt.close()


# ============================================================================
# 7. DETECCIÓN PRELIMINAR DE OUTLIERS (IQR)
# ============================================================================
print("\n")
print("7. DETECCIÓN PRELIMINAR DE OUTLIERS (método IQR)")


print(
    f"\n  {'Variable':30s} {'Q1':>10s} {'Q3':>10s} {'IQR':>10s} "
    f"{'Lím.Inf':>10s} {'Lím.Sup':>10s} {'Outliers':>10s} {'%':>8s}"
)


outlier_counts = {}
for col in NUM_COLS:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((s < lower) | (s > upper)).sum()
    pct = n_outliers / len(s) * 100
    outlier_counts[col] = n_outliers
    print(
        f"  {col:30s} {q1:10.2f} {q3:10.2f} {iqr:10.2f} "
        f"{lower:10.2f} {upper:10.2f} {n_outliers:10,} {pct:7.2f}%"
    )

# Boxplots comparativos para outliers
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    if i >= len(axes):
        break
    s = df[col].dropna()
    bp = axes[i].boxplot(
        s, vert=True, patch_artist=True, boxprops=dict(facecolor="#9b59b6", alpha=0.6)
    )
    axes[i].set_title(f"Outliers: {col}")
    axes[i].set_ylabel(col)

plt.suptitle("Detección de Outliers (IQR) - Variables Numéricas", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "06_outliers_boxplots.png"))
plt.close()


# ============================================================================
# 8. RELACIÓN ENTRE VARIABLES CATEGÓRICAS Y EL TARGET
# ============================================================================
print("\n")
print("8. RELACIÓN ENTRE VARIABLES CATEGÓRICAS Y EL TARGET")


for col in CAT_ORDINAL_COLS + CAT_NOMINAL_COLS:
    ct = pd.crosstab(df[col], df[TARGET_COL], normalize="index") * 100
    print(f"\n  {col} vs {TARGET_COL} (% fila):")
    print(ct.round(1).to_string())

# Gráficos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(CAT_ORDINAL_COLS + CAT_NOMINAL_COLS):
    ct = pd.crosstab(df[col], df[TARGET_COL])
    ct.plot(
        kind="bar", stacked=True, ax=axes[i], color=["#3498db", "#e74c3c"], alpha=0.85
    )
    axes[i].set_title(f"{col} vs {TARGET_COL}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frecuencia")
    axes[i].legend(title=TARGET_COL)
    axes[i].tick_params(axis="x", rotation=45)

plt.suptitle("Variables Categóricas vs Variable Objetivo", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07_categoricas_vs_target.png"))
plt.close()


# ============================================================================
# 9. DISTRIBUCIÓN DE NUMÉRICAS POR GRUPO (addicted_label)
# ============================================================================
print("\n")
print("9. DISTRIBUCIÓN DE NUMÉRICAS POR GRUPO (addicted_label)")


for col in NUM_COLS:
    g0 = df.loc[df[TARGET_COL] == 0, col].describe()
    g1 = df.loc[df[TARGET_COL] == 1, col].describe()
    comp = pd.DataFrame({"No adicto (0)": g0, "Adicto (1)": g1})
    print(f"\n  {col}:")
    print(comp.round(3).to_string())

# Violin plots
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    if i >= len(axes):
        break
    sns.violinplot(
        data=df,
        x=TARGET_COL,
        y=col,
        ax=axes[i],
        palette=["#3498db", "#e74c3c"],
        inner="quartile",
    )
    axes[i].set_title(f"{col} por {TARGET_COL}")

plt.suptitle("Distribuciones Numéricas por Grupo de Adicción", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "08_violines_por_grupo.png"))
plt.close()
print("  Gráfico guardado: 08_violines_por_grupo.png")

# ============================================================================
# 10. DUPLICADOS Y RESUMEN DE CALIDAD
# ============================================================================
print("\n")
print("10. DUPLICADOS Y RESUMEN DE CALIDAD")


n_dup_id = df["transaction_id"].duplicated().sum()
n_dup_full = df.duplicated().sum()
print(f"  transaction_id duplicados:       {n_dup_id:,}")
print(f"  Filas completamente duplicadas:  {n_dup_full:,}")

total_cells = df.size
null_total = df.isnull().sum().sum()
print(f"\n  Total de celdas:            {total_cells:,}")
print(
    f"  Celdas con NaN:             {null_total:,} ({null_total / total_cells * 100:.2f}%)"
)

print("\n  Clasificación de variables por calidad:")
for col in df.columns:
    n_null = df[col].isnull().sum()
    pct_bad = n_null / len(df) * 100
    if pct_bad == 0:
        quality = "Limpia"
    elif pct_bad < 5:
        quality = "Aceptable"
    elif pct_bad < 15:
        quality = "Requiere atención"
    else:
        quality = "Problemática"
    print(f"    {col:30s}  NaN={n_null:5,}  ({pct_bad:5.1f}%)  {quality}")

# ============================================================================
# 11. PAIRPLOT DE VARIABLES CLAVE (muestra)
# ============================================================================
print("\n  Generamos pairplot ")
key_vars = [
    "daily_screen_time_hours",
    "social_media_hours",
    "sleep_hours",
    "stress_level",
    TARGET_COL,
]
sample = df[key_vars].sample(min(2000, len(df)), random_state=42)
sample[TARGET_COL] = sample[TARGET_COL].astype(str)

g = sns.pairplot(
    sample,
    hue=TARGET_COL,
    palette={"0": "#3498db", "1": "#e74c3c"},
    diag_kind="kde",
    plot_kws={"alpha": 0.3, "s": 12},
)
g.figure.suptitle(
    "Pairplot - Variables clave por grupo de adicción", fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "09_pairplot.png"))
plt.close()
