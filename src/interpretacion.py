"""

interpretacion.py  -  Interpretacion y Conclusiones
Dataset: Smartphone Addiction Prediction Data
Trabajo Final - Mineria de Datos - Grado en Matematicas

Realizamos:
  1. Analisis de residuos para el modelo lineal (Regresion Logistica)
  2. Importancia de variables en modelos de caja negra (SVM, Random Forest)
  3. Analisis critico: limitaciones y sobreajuste

"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
DATA_PATH = os.path.join(BASE_DIR, "data", "data_clean.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")
os.makedirs(IMG_DIR, exist_ok=True)

# Carga de datos
df = pd.read_csv(DATA_PATH)
TARGET = "addicted_label"
FEATURES = ["daily_screen_time_hours", "social_media_hours", "sleep_hours"]

X = df[FEATURES].values
y = df[TARGET].values

## Usamos == como separador entre secciones para organizar el codigo que si no me lio

# ============================================================================
# 0. REENTRENAR MODELOS (misma configuracion que modelizacion.py)
# ============================================================================
print("\n")
print("0. REENTRENANDO MODELOS")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Regresion Logistica
best_lr = LogisticRegression(
    C=0.1,
    penalty="l2",
    solver="saga",
    class_weight=None,
    max_iter=5000,
    random_state=42,
)
best_lr.fit(X_train_sc, y_train)
y_prob_lr = best_lr.predict_proba(X_test_sc)[:, 1]
print("  Regresion Logistica entrenada")

# SVM
best_svm = SVC(
    C=100, gamma=1, kernel="rbf", class_weight=None, probability=True, random_state=42
)
best_svm.fit(X_train_sc, y_train)
print("  SVM entrenado")

# Random Forest
best_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    class_weight=None,
    random_state=42,
)
best_rf.fit(X_train, y_train)
print("  Random Forest entrenado")

# ============================================================================
# 1. ANALISIS DE RESIDUOS - REGRESION LOGISTICA
# ============================================================================
print("\n")
print("1. ANALISIS DE RESIDUOS - REGRESION LOGISTICA")

# Ajustar modelo con statsmodels para obtener diagnosticos completos
X_train_sm = sm.add_constant(X_train_sc)
X_test_sm = sm.add_constant(X_test_sc)

logit_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
print(logit_model.summary())

# Probabilidades predichas en test
p_hat = logit_model.predict(X_test_sm)
p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)  # Evitar log(0)

# Residuos de devianza: d_i = sign(y-p) * sqrt(-2 * [y*log(p) + (1-y)*log(1-p)])
sign = np.where(y_test == 1, 1, -1)
dev_component = -2 * (y_test * np.log(p_hat) + (1 - y_test) * np.log(1 - p_hat))
deviance_residuals = sign * np.sqrt(dev_component)

# Residuos de Pearson: (y - p) / sqrt(p * (1-p))
pearson_residuals = (y_test - p_hat) / np.sqrt(p_hat * (1 - p_hat))

# Valores ajustados (linear predictor)
linear_pred = X_test_sm @ logit_model.params

print("\n  Residuos de devianza:")
print(f"    Media:     {deviance_residuals.mean():.4f}")
print(f"    Desv. tip: {deviance_residuals.std():.4f}")
print(f"    Min:       {deviance_residuals.min():.4f}")
print(f"    Max:       {deviance_residuals.max():.4f}")

print("\n  Residuos de Pearson:")
print(f"    Media:     {pearson_residuals.mean():.4f}")
print(f"    Desv. tip: {pearson_residuals.std():.4f}")
print(f"    Min:       {pearson_residuals.min():.4f}")
print(f"    Max:       {pearson_residuals.max():.4f}")

# Test de Hosmer-Lemeshow (bondad de ajuste)
n_groups = 10
sorted_idx = np.argsort(p_hat)
groups = np.array_split(sorted_idx, n_groups)

hl_obs = []
hl_exp = []
for g in groups:
    hl_obs.append(y_test[g].sum())
    hl_exp.append(p_hat[g].sum())

hl_obs = np.array(hl_obs)
hl_exp = np.array(hl_exp)
n_g = np.array([len(g) for g in groups])

# Estadistico HL
hl_stat = np.sum((hl_obs - hl_exp) ** 2 / (hl_exp * (1 - hl_exp / n_g) + 1e-10))
hl_pval = 1 - stats.chi2.cdf(hl_stat, df=n_groups - 2)

print("\n  Test de Hosmer-Lemeshow:")
print(f"    Estadistico: {hl_stat:.4f}")
print(f"    p-valor:     {hl_pval:.4f}")
if hl_pval > 0.05:
    print("    Conclusion: No se rechaza H0 -> el modelo ajusta bien")
else:
    print("    Conclusion: Se rechaza H0 -> posible falta de ajuste")

# -- Graficos de residuos --

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1.1 Residuos de devianza vs probabilidad predicha
ax = axes[0, 0]
ax.scatter(p_hat, deviance_residuals, alpha=0.3, s=8, c="#3498db")
ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
ax.axhline(y=2, color="gray", linestyle=":", alpha=0.5)
ax.axhline(y=-2, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Probabilidad predicha")
ax.set_ylabel("Residuos de devianza")
ax.set_title("Residuos de Devianza vs Prob. Predicha")
ax.grid(alpha=0.2)

# 1.2 Residuos de Pearson vs predictor lineal
ax = axes[0, 1]
ax.scatter(linear_pred, pearson_residuals, alpha=0.3, s=8, c="#e74c3c")
ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
ax.axhline(y=2, color="gray", linestyle=":", alpha=0.5)
ax.axhline(y=-2, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Predictor lineal (logit)")
ax.set_ylabel("Residuos de Pearson")
ax.set_title("Residuos de Pearson vs Predictor Lineal")
ax.grid(alpha=0.2)

# 1.3 Q-Q plot de residuos de devianza
ax = axes[1, 0]
sorted_res = np.sort(deviance_residuals)
n = len(sorted_res)
theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
ax.scatter(theoretical, sorted_res, alpha=0.3, s=8, c="#2ecc71")
# Linea de referencia
q25, q75 = np.percentile(sorted_res, [25, 75])
t25, t75 = stats.norm.ppf([0.25, 0.75])
slope = (q75 - q25) / (t75 - t25)
intercept = q25 - slope * t25
x_line = np.array([theoretical.min(), theoretical.max()])
ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.5)
ax.set_xlabel("Cuantiles teoricos (Normal)")
ax.set_ylabel("Cuantiles de residuos de devianza")
ax.set_title("Q-Q Plot - Residuos de Devianza")
ax.grid(alpha=0.2)

# 1.4 Histograma de residuos de devianza
ax = axes[1, 1]
ax.hist(
    deviance_residuals,
    bins=40,
    color="#3498db",
    alpha=0.7,
    edgecolor="white",
    density=True,
)
# Curva normal de referencia
x_norm = np.linspace(deviance_residuals.min(), deviance_residuals.max(), 100)
ax.plot(
    x_norm,
    stats.norm.pdf(x_norm, deviance_residuals.mean(), deviance_residuals.std()),
    "r-",
    linewidth=2,
    label="Normal ajustada",
)
ax.set_xlabel("Residuos de devianza")
ax.set_ylabel("Densidad")
ax.set_title("Distribucion de Residuos de Devianza")
ax.legend()
ax.grid(alpha=0.2)

plt.suptitle("Diagnostico de Residuos - Regresion Logistica", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "28_residuos_logistica.png"))
plt.close()

# Hosmer-Lemeshow grafico
fig, ax = plt.subplots(figsize=(8, 5))
x_groups = np.arange(1, n_groups + 1)
width = 0.35
ax.bar(
    x_groups - width / 2, hl_obs, width, label="Observados", color="#3498db", alpha=0.85
)
ax.bar(
    x_groups + width / 2, hl_exp, width, label="Esperados", color="#e74c3c", alpha=0.85
)
ax.set_xlabel("Decil de probabilidad")
ax.set_ylabel("Numero de positivos")
ax.set_title(f"Test de Hosmer-Lemeshow (Chi2={hl_stat:.2f}, p={hl_pval:.4f})")
ax.set_xticks(x_groups)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "29_hosmer_lemeshow.png"))
plt.close()

# ============================================================================
# 2. IMPORTANCIA DE VARIABLES - MODELOS DE CAJA NEGRA
# ============================================================================
print("\n")
print("2. IMPORTANCIA DE VARIABLES - MODELOS DE CAJA NEGRA")

# 2.1 Importancia Gini (Random Forest - intrinseca)
gini_imp = best_rf.feature_importances_
print("\n  Importancia Gini (Random Forest):")
for feat, imp in sorted(zip(FEATURES, gini_imp), key=lambda x: -x[1]):
    print(f"    {feat:30s}  {imp:.4f}")

# 2.2 Permutation Importance (model-agnostic) para los 3 modelos
print("\n  Permutation Importance (30 repeticiones):")

models_for_pi = {
    "Regresion Logistica": (best_lr, X_test_sc),
    "SVM (RBF)": (best_svm, X_test_sc),
    "Random Forest": (best_rf, X_test),
}

perm_results = {}
for name, (model, X_pi) in models_for_pi.items():
    result = permutation_importance(
        model, X_pi, y_test, n_repeats=30, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    perm_results[name] = result
    print(f"\n  {name}:")
    for feat, mean, std in sorted(
        zip(FEATURES, result.importances_mean, result.importances_std),
        key=lambda x: -x[1],
    ):
        print(f"    {feat:30s}  {mean:.4f} (+/- {std:.4f})")

# Grafico comparativo de permutation importance
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(FEATURES))
width = 0.25
colors = ["#3498db", "#e74c3c", "#2ecc71"]

for i, (name, result) in enumerate(perm_results.items()):
    sorted_idx = np.argsort(result.importances_mean)[::-1]
    means = result.importances_mean
    stds = result.importances_std
    bars = ax.bar(
        x + i * width,
        means,
        width,
        yerr=stds,
        label=name,
        color=colors[i],
        alpha=0.85,
        capsize=3,
    )
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

ax.set_xticks(x + width)
ax.set_xticklabels(FEATURES, fontsize=9)
ax.set_ylabel("Permutation Importance (AUC-ROC)")
ax.set_title("Importancia de Variables por Modelo (Permutation Importance)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "30_permutation_importance.png"))
plt.close()

# Comparativa Gini vs Permutation para Random Forest
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Gini
ax = axes[0]
sorted_idx = np.argsort(gini_imp)
ax.barh(
    np.array(FEATURES)[sorted_idx],
    gini_imp[sorted_idx],
    color=["#2ecc71", "#3498db", "#e74c3c"][: len(FEATURES)],
)
ax.set_xlabel("Importancia (Gini)")
ax.set_title("Random Forest - Importancia Gini")
for i, idx in enumerate(sorted_idx):
    ax.text(gini_imp[idx] + 0.005, i, f"{gini_imp[idx]:.3f}", va="center", fontsize=9)

# Permutation
ax = axes[1]
pi_rf = perm_results["Random Forest"]
sorted_idx = np.argsort(pi_rf.importances_mean)
ax.barh(
    np.array(FEATURES)[sorted_idx],
    pi_rf.importances_mean[sorted_idx],
    xerr=pi_rf.importances_std[sorted_idx],
    color=["#2ecc71", "#3498db", "#e74c3c"][: len(FEATURES)],
    capsize=3,
)
ax.set_xlabel("Permutation Importance (AUC-ROC)")
ax.set_title("Random Forest - Permutation Importance")
for i, idx in enumerate(sorted_idx):
    ax.text(
        pi_rf.importances_mean[idx] + 0.005,
        i,
        f"{pi_rf.importances_mean[idx]:.3f}",
        va="center",
        fontsize=9,
    )

plt.suptitle("Importancia de Variables - Random Forest", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "31_gini_vs_permutation.png"))
plt.close()

# ============================================================================
# 3. COEFICIENTES DEL MODELO LOGISTICO (interpretabilidad)
# ============================================================================
print("\n")
print("3. COEFICIENTES DEL MODELO LOGISTICO")

# Odds ratios
coefs = logit_model.params[1:]  # sin intercepto
odds_ratios = np.exp(coefs)
ci = logit_model.conf_int()[1:]
odds_ci_low = np.exp(ci[:, 0])
odds_ci_high = np.exp(ci[:, 1])

print("\n  Odds Ratios (variables estandarizadas):")
for feat, or_val, lo, hi in zip(FEATURES, odds_ratios, odds_ci_low, odds_ci_high):
    print(f"    {feat:30s}  OR={or_val:.4f}  IC95%=[{lo:.4f}, {hi:.4f}]")

# Grafico de odds ratios
fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(FEATURES))
ax.barh(y_pos, odds_ratios, color="#3498db", alpha=0.85, height=0.5)
ax.errorbar(
    odds_ratios,
    y_pos,
    xerr=[odds_ratios - odds_ci_low, odds_ci_high - odds_ratios],
    fmt="none",
    color="black",
    capsize=5,
)
ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="OR = 1 (sin efecto)")
ax.set_yticks(y_pos)
ax.set_yticklabels(FEATURES)
ax.set_xlabel("Odds Ratio (escala estandarizada)")
ax.set_title("Odds Ratios - Regresion Logistica")
ax.legend()
ax.grid(axis="x", alpha=0.3)

for i, (or_val, lo, hi) in enumerate(zip(odds_ratios, odds_ci_low, odds_ci_high)):
    ax.text(or_val + 0.3, i, f"OR={or_val:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "32_odds_ratios.png"))
plt.close()
