"""

  modelizacion.py  -  Modelizacion Supervisada y Contraste
  Dataset: Smartphone Addiction Prediction Data
  Trabajo Final - Mineria de Datos - Grado en Matematicas

  Modelos implementados:
    1. Regresion Logistica (Baseline - modelo lineal)
    2. SVM con kernel RBF (Modelo flexible - relaciones no lineales)
    3. Random Forest (Metodo de agregacion / Ensemble)

  Metodologia:
    - Train/Test split estratificado (80/20)
    - Validacion Cruzada estratificada 5-fold para ajuste de hiperparametros
    - Metricas: Accuracy, Precision, Recall, F1-Score, AUC-ROC
    - Analisis del compromiso sesgo-varianza
    - Comparativa final entre modelos

"""

import os
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
# Parametros graficos
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data_clean.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")
os.makedirs(IMG_DIR, exist_ok=True)

# Carga de datos
df = pd.read_csv(DATA_PATH)
TARGET = "addicted_label"
FEATURES = ["daily_screen_time_hours", "social_media_hours","sleep_hours"]

X = df[FEATURES].values
y = df[TARGET].values

## Usamos == como separador entre secciones para organizar el código que si no me lío

# ============================================================================
# 1. DIVISION TRAIN / TEST
# ============================================================================
print("\n")
print("1. DIVISION TRAIN / TEST (80/20 estratificado)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# Estandarizacion
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Validacion cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# 2. REGRESION LOGISTICA (BASELINE)
# ============================================================================
print("\n")
print("2. REGRESION LOGISTICA (BASELINE)")

# Parametros
param_grid_lr = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["saga"],
    "class_weight": [None, "balanced"],
}

gs_lr = GridSearchCV(
    LogisticRegression(max_iter=5000, random_state=42),
    param_grid_lr,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    return_train_score=True,
)
gs_lr.fit(X_train_sc, y_train)

best_lr = gs_lr.best_estimator_
print(f"\n  Mejores hiperparametros: {gs_lr.best_params_}")
print(f"  AUC-ROC CV (train): {gs_lr.cv_results_['mean_train_score'][gs_lr.best_index_]:.4f}")
print(f"  AUC-ROC CV (valid): {gs_lr.best_score_:.4f}")

# Evaluacion en test
y_pred_lr = best_lr.predict(X_test_sc)
y_prob_lr = best_lr.predict_proba(X_test_sc)[:, 1]

print("\n  Metricas en Test")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred_lr):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob_lr):.4f}")

# Coeficientes del modelo
print("\n  Coeficientes")
for feat, coef in zip(FEATURES, best_lr.coef_[0]):
    print(f"    {feat:30s}  {coef:+.4f}")
print(f"    {'(intercepto)':30s}  {best_lr.intercept_[0]:+.4f}")

# ============================================================================
# 3. SVM CON KERNEL RBF (MODELO FLEXIBLE)
# ============================================================================
print("\n")
print("3. SVM CON KERNEL RBF (MODELO FLEXIBLE)")

# Parametros
param_grid_svm = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
    "class_weight": [None, "balanced"],
}

gs_svm = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=42),
    param_grid_svm,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    return_train_score=True,
)
gs_svm.fit(X_train_sc, y_train)

best_svm = gs_svm.best_estimator_
print(f"\n  Mejores hiperparametros: {gs_svm.best_params_}")
print(f"  AUC-ROC CV (train): {gs_svm.cv_results_['mean_train_score'][gs_svm.best_index_]:.4f}")
print(f"  AUC-ROC CV (valid): {gs_svm.best_score_:.4f}")

# Evaluacion en test
y_pred_svm = best_svm.predict(X_test_sc)
y_prob_svm = best_svm.predict_proba(X_test_sc)[:, 1]

print("\n  Metricas en Test")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_svm):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred_svm):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob_svm):.4f}")

# Vectores soporte
print(f"\n  Vectores soporte: {best_svm.n_support_} (total={sum(best_svm.n_support_)})")

# ============================================================================
# 4. RANDOM FOREST (ENSEMBLE)
# ============================================================================
print("\n")
print("4. RANDOM FOREST (ENSEMBLE)")

# Parametros
param_grid_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"],
}

gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    return_train_score=True,
)
# Random Forest no requiere estandarizacion, pero lo usamos para consistencia
gs_rf.fit(X_train, y_train)

best_rf = gs_rf.best_estimator_
print(f"\n  Mejores hiperparametros: {gs_rf.best_params_}")
print(f"  AUC-ROC CV (train): {gs_rf.cv_results_['mean_train_score'][gs_rf.best_index_]:.4f}")
print(f"  AUC-ROC CV (valid): {gs_rf.best_score_:.4f}")

# Evaluacion en test
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

print("\n Metricas en Test")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob_rf):.4f}")

# Importancia de variables
print("\n  Importancia de variables")
importances = best_rf.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    print(f"    {feat:30s}  {imp:.4f}")

# ============================================================================
# 5. COMPARATIVA FINAL
# ============================================================================
print("\n")
print("5. COMPARATIVA FINAL DE MODELOS")


models_results = {
    "Regresion Logistica": {
        "y_pred": y_pred_lr, "y_prob": y_prob_lr,
        "best_params": gs_lr.best_params_,
        "cv_train_auc": gs_lr.cv_results_["mean_train_score"][gs_lr.best_index_],
        "cv_valid_auc": gs_lr.best_score_,
    },
    "SVM (RBF)": {
        "y_pred": y_pred_svm, "y_prob": y_prob_svm,
        "best_params": gs_svm.best_params_,
        "cv_train_auc": gs_svm.cv_results_["mean_train_score"][gs_svm.best_index_],
        "cv_valid_auc": gs_svm.best_score_,
    },
    "Random Forest": {
        "y_pred": y_pred_rf, "y_prob": y_prob_rf,
        "best_params": gs_rf.best_params_,
        "cv_train_auc": gs_rf.cv_results_["mean_train_score"][gs_rf.best_index_],
        "cv_valid_auc": gs_rf.best_score_,
    },
}

header = f"  {'Modelo':<25s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'AUC-ROC':>9s}"
print(header)
print("  " + "-" * (len(header) - 2))

for name, res in models_results.items():
    acc = accuracy_score(y_test, res["y_pred"])
    prec = precision_score(y_test, res["y_pred"])
    rec = recall_score(y_test, res["y_pred"])
    f1 = f1_score(y_test, res["y_pred"])
    auc = roc_auc_score(y_test, res["y_prob"])
    print(f"  {name:<25s} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {auc:>9.4f}")

# Analisis sesgo-varianza
print("\n  Analisis Sesgo-Varianza (AUC-ROC)")
header2 = f"  {'Modelo':<25s} {'CV Train':>10s} {'CV Valid':>10s} {'Gap':>8s} {'Diagnostico'}"
print(header2)

for name, res in models_results.items():
    gap = res["cv_train_auc"] - res["cv_valid_auc"]
    if gap < 0.02:
        diag = "Buen ajuste"
    elif gap < 0.05:
        diag = "Ligero sobreajuste"
    else:
        diag = "Sobreajuste"
    print(f"  {name:<25s} {res['cv_train_auc']:>10.4f} {res['cv_valid_auc']:>10.4f} {gap:>8.4f}  {diag}")


# ============================================================================
# 6. GRAFICOS
# ============================================================================
print("\n")
print("6. GRAFICOS")


COLORS = {"Regresion Logistica": "#3498db",
           "SVM (RBF)": "#e74c3c",
           "Random Forest": "#2ecc71"}

# Curvas ROC 
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in models_results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    auc_val = roc_auc_score(y_test, res["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
            color=COLORS[name], linewidth=2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Azar (AUC=0.500)")
ax.set_xlabel("False Positive Rate (1 - Especificidad)")
ax.set_ylabel("True Positive Rate (Sensibilidad)")
ax.set_title("Curvas ROC - Comparativa de Modelos")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "15_curvas_roc.png"))
plt.close()

# Matrices de confusion
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, res) in zip(axes, models_results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["No adicto", "Adicto"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)
plt.suptitle("Matrices de Confusion", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "16_matrices_confusion.png"))
plt.close()


# Comparativa de metricas (barplot)
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
metrics_data = {}
for name, res in models_results.items():
    metrics_data[name] = [
        accuracy_score(y_test, res["y_pred"]),
        precision_score(y_test, res["y_pred"]),
        recall_score(y_test, res["y_pred"]),
        f1_score(y_test, res["y_pred"]),
        roc_auc_score(y_test, res["y_prob"]),
    ]

x = np.arange(len(metrics_names))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, vals) in enumerate(metrics_data.items()):
    bars = ax.bar(x + i * width, vals, width, label=name, color=COLORS[name], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Valor")
ax.set_title("Comparativa de Metricas por Modelo")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "17_comparativa_metricas.png"))
plt.close()


# Importancia de variables (Random Forest)
fig, ax = plt.subplots(figsize=(8, 4))
sorted_idx = np.argsort(importances)
ax.barh(np.array(FEATURES)[sorted_idx], importances[sorted_idx],
        color=["#2ecc71", "#3498db", "#e74c3c"][:len(FEATURES)])
ax.set_xlabel("Importancia (Gini)")
ax.set_title("Importancia de Variables - Random Forest")
for i, (idx) in enumerate(sorted_idx):
    ax.text(importances[idx] + 0.005, i, f"{importances[idx]:.3f}",
            va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "18_importancia_rf.png"))
plt.close()


# Learning curves (sesgo-varianza)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
models_for_lc = [
    ("Regresion Logistica", LogisticRegression(**gs_lr.best_params_, max_iter=5000, random_state=42)),
    ("SVM (RBF)", SVC(**gs_svm.best_params_, kernel="rbf", probability=True, random_state=42)),
    ("Random Forest", RandomForestClassifier(**gs_rf.best_params_, random_state=42)),
]

for ax, (name, model) in zip(axes, models_for_lc):
    # SVM y LR usan datos escalados
    X_lc = X_train_sc if name != "Random Forest" else X_train

    train_sizes, train_scores, val_scores = learning_curve(
        model, X_lc, y_train, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=COLORS[name])
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="gray")
    ax.plot(train_sizes, train_mean, "o-", color=COLORS[name], label="Train", markersize=4)
    ax.plot(train_sizes, val_mean, "o-", color="gray", label="Validacion", markersize=4)
    ax.set_title(name)
    ax.set_xlabel("Tamano de entrenamiento")
    ax.set_ylabel("AUC-ROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.05)

plt.suptitle("Learning Curves - Analisis Sesgo-Varianza", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "19_learning_curves.png"))
plt.close()


# Sesgo-varianza barplot
fig, ax = plt.subplots(figsize=(8, 5))
names_list = list(models_results.keys())
cv_train = [models_results[n]["cv_train_auc"] for n in names_list]
cv_valid = [models_results[n]["cv_valid_auc"] for n in names_list]

x = np.arange(len(names_list))
width = 0.35
bars1 = ax.bar(x - width/2, cv_train, width, label="CV Train (AUC)", color="#3498db", alpha=0.8)
bars2 = ax.bar(x + width/2, cv_valid, width, label="CV Validation (AUC)", color="#e67e22", alpha=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(names_list)
ax.set_ylabel("AUC-ROC")
ax.set_title("Compromiso Sesgo-Varianza: Train vs Validacion")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.7, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "20_sesgo_varianza.png"))
plt.close()



# Distribucion de probabilidades predichas
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, res) in zip(axes, models_results.items()):
    for label, label_name in [(0, "No adicto"), (1, "Adicto")]:
        mask = y_test == label
        ax.hist(res["y_prob"][mask], bins=30, alpha=0.6, label=label_name,
                color="#3498db" if label == 0 else "#e74c3c")
    ax.set_xlabel("Probabilidad predicha (clase 1)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.axvline(x=0.5, color="k", linestyle="--", alpha=0.5)
plt.suptitle("Distribucion de Probabilidades Predichas", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "21_distribucion_probabilidades.png"))
plt.close()



