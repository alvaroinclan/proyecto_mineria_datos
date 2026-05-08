"""

 ingenieria_variables.py  -  Selección de Variables y Preparación
 Dataset: Smartphone Addiction Prediction Data
 Trabajo Final - Minería de Datos - Grado en Matemáticas

 Realizamos:
   1. Limpieza previa (eliminar IDs, leakage, colinealidad)
   2. Codificación de variables categóricas
   3. Selección de variables mediante Stepwise (forward) con AIC
   4. Exportación del dataset final listo para modelización

"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data base.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "data_clean.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")
os.makedirs(IMG_DIR, exist_ok=True)

# Carga
df = pd.read_csv(DATA_PATH)


## Usamos == como separador entre secciones para organizar el código que si no me lío

# ============================================================================
# 1. LIMPIEZA PREVIA (decisiones del EDA y tratamiento de faltantes)
# ============================================================================
print("\n")
print("1. LIMPIEZA PREVIA")


# Eliminar identificadores
df = df.drop(columns=["transaction_id", "user_id"])
print("Eliminadas: transaction_id, user_id (identificadores)")

# Eliminar addiction_level (data leakage)
df = df.drop(columns=["addiction_level"])
print("Eliminada: addiction_level (data leakage)")

# Eliminar weekend_screen_time (colinealidad r=0.964 con daily_screen_time)
df = df.drop(columns=["weekend_screen_time"])
print("Eliminada: weekend_screen_time (colinealidad)")

print(f"\n Dimensiones tras limpieza: {df.shape}")

# ============================================================================
# 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================================
print("\n")
print("2. CODIFICACION DE VARIABLES CATEGORICAS")


# stress_level: ordinal
stress_map = {"Low": 0, "Medium": 1, "High": 2}
df["stress_level"] = df["stress_level"].map(stress_map)
print(f"stress_level: ordinal: {stress_map}")

# academic_work_impact: binaria
impact_map = {"No": 0, "Yes": 1}
df["academic_work_impact"] = df["academic_work_impact"].map(impact_map)
print(f"academic_work_impact: binaria: {impact_map}")

# gender: one-hot (drop_first para evitar multicolinealidad)
gender_dummies = pd.get_dummies(df["gender"], prefix="gender", drop_first=True)
gender_dummies = gender_dummies.astype(int)
df = pd.concat([df.drop(columns=["gender"]), gender_dummies], axis=1)
print(f"gender: one-hot: {list(gender_dummies.columns)}")

# Separar predictoras y target
TARGET = "addicted_label"
y = df[TARGET]
X = df.drop(columns=[TARGET])

print(f"\nVariables predictoras ({X.shape[1]}):")
for col in X.columns:
    print(f"    · {col}")
print(f"Variable objetivo: {TARGET}")

# ============================================================================
# 3. SELECCIÓN STEPWISE (FORWARD) CON AIC
# ============================================================================
print("\n")
print("3. SELECCION STEPWISE (FORWARD) CON AIC")


print("""
  Metodo: Forward Stepwise Selection
  Criterio: AIC (Akaike Information Criterion)
  Modelo base: Regresion Logistica (Logit)

  En cada paso se anade la variable que mas reduce el AIC.
  Se detiene cuando ninguna variable adicional mejora el AIC.
""")


def stepwise_forward_aic(X, y):
    """Seleccion forward stepwise basada en AIC con regresion logistica."""
    remaining = list(X.columns)
    selected = []
    # Modelo nulo (solo intercepto)
    model_null = sm.Logit(y, sm.add_constant(pd.DataFrame(index=X.index))).fit(
        disp=0
    )
    current_aic = model_null.aic
    print(f"  Paso 0 (intercepto): AIC = {current_aic:.2f}")

    step = 0
    while remaining:
        step += 1
        aic_candidates = {}
        for var in remaining:
            X_test = sm.add_constant(X[selected + [var]])
            try:
                model = sm.Logit(y, X_test).fit(disp=0, maxiter=100)
                aic_candidates[var] = model.aic
            except Exception:
                continue

        if not aic_candidates:
            break

        best_var = min(aic_candidates, key=aic_candidates.get)
        best_aic = aic_candidates[best_var]

        if best_aic < current_aic:
            selected.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            print(f"  Paso {step}: + {best_var:30s}  AIC = {best_aic:.2f}")
        else:
            print(f"  Paso {step}: ninguna variable mejora el AIC -> STOP")
            break

    return selected, current_aic


selected_vars, final_aic = stepwise_forward_aic(X, y)

print(f"\n  Variables seleccionadas ({len(selected_vars)}):")
for i, var in enumerate(selected_vars, 1):
    print(f"    {i}. {var}")
print(f"  AIC final: {final_aic:.2f}")

# Variables descartadas
dropped_vars = [v for v in X.columns if v not in selected_vars]
print(f"\n  Variables descartadas ({len(dropped_vars)}):")
for var in dropped_vars:
    print(f"    · {var}")

# ============================================================================
# 4. MODELO FINAL CON VARIABLES SELECCIONADAS
# ============================================================================
print("\n")
print("4. RESUMEN DEL MODELO LOGISTICO CON VARIABLES SELECCIONADAS")


X_final = sm.add_constant(X[selected_vars])
model_final = sm.Logit(y, X_final).fit(disp=0)
print(model_final.summary())

# ============================================================================
# 5. GRÁFICOS
# ============================================================================
print("\n")
print("5. GRAFICOS")


# Importancia de variables (coeficientes del modelo logístico)
fig, ax = plt.subplots(figsize=(10, 5))
coefs = model_final.params.drop("const")
pvals = model_final.pvalues.drop("const")
colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in pvals]

coefs_sorted = coefs.abs().sort_values(ascending=True)
bars = ax.barh(coefs_sorted.index, coefs_sorted.values,
               color=[colors[list(coefs.index).index(v)]
                      for v in coefs_sorted.index])
ax.set_xlabel("|Coeficiente| (Logit)")
ax.set_title("Importancia de variables seleccionadas (Stepwise Forward)")
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ecc71", label="p < 0.05"),
                   Patch(facecolor="#e74c3c", label="p >= 0.05")]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "12_importancia_stepwise.png"))
plt.close()


# Evolución del AIC durante el stepwise (re-calcular para graficar)
print("Calculamos evolucion del AIC")
aic_history = []
model_null = sm.Logit(y, sm.add_constant(pd.DataFrame(index=X.index))).fit(disp=0)
aic_history.append(("(intercepto)", model_null.aic))

cumulative = []
for var in selected_vars:
    cumulative.append(var)
    X_tmp = sm.add_constant(X[cumulative])
    model_tmp = sm.Logit(y, X_tmp).fit(disp=0)
    aic_history.append((var, model_tmp.aic))

fig, ax = plt.subplots(figsize=(10, 5))
labels = [h[0] for h in aic_history]
aics = [h[1] for h in aic_history]
ax.plot(range(len(aics)), aics, "o-", color="#3498db", linewidth=2, markersize=8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("AIC")
ax.set_xlabel("Variable added")
ax.set_title("AIC Evolution - Stepwise Forward")
ax.grid(axis="y", alpha=0.3)

for i, (label, aic) in enumerate(aic_history):
    ax.annotate(f"{aic:.0f}", (i, aic), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "13_evolucion_aic.png"))
plt.close()


# Correlaciones del dataset final
fig, ax = plt.subplots(figsize=(8, 6))
df_final_corr = X[selected_vars].copy()
df_final_corr[TARGET] = y
corr = df_final_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlations -- Selected Variables")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "14_correlaciones_final.png"))
plt.close()


# ============================================================================
# 6. EXPORTACIÓN DEL DATASET LIMPIO
# ============================================================================
df_export = X[selected_vars].copy()
df_export[TARGET] = y
df_export.to_csv(CLEAN_PATH, index=False)

