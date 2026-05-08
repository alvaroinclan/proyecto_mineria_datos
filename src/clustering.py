"""

clustering.py  -  Aprendizaje No Supervisado
Dataset: Smartphone Addiction Prediction Data
Trabajo Final - Mineria de Datos - Grado en Matematicas

Realizamos:
  1. Clustering Jerarquico Aglomerativo
  2. Seleccion del numero optimo de clusters (dendrograma + metricas)
  3. Perfilado de los clusters obtenidos
  4. Analisis de clusters obtenidos

"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

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
# 1. ESTANDARIZACION
# ============================================================================
print("\n")
print("1. ESTANDARIZACION")

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

print(f"  Variables: {FEATURES}")
print(f"  Observaciones: {X_sc.shape[0]}")

# ============================================================================
# 2. DENDROGRAMA Y SELECCION DE METODO DE ENLACE
# ============================================================================
print("\n")
print("2. DENDROGRAMA")

# Usamos una submuestra para el dendrograma (7500 es demasiado para visualizar)
np.random.seed(42)
sample_idx = np.random.choice(len(X_sc), size=1500, replace=False)
X_sample = X_sc[sample_idx]
y_sample = y[sample_idx]

# Comparar metodos de enlace
methods = ["ward", "complete", "average"]
linkage_matrices = {}

for method in methods:
    Z = linkage(X_sample, method=method, metric="euclidean")
    linkage_matrices[method] = Z

# Dendrograma con metodo Ward (el mas comun para clusters compactos)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, method in zip(axes, methods):
    Z = linkage_matrices[method]
    dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_title(f"Metodo: {method.capitalize()}")
    ax.set_xlabel("Observaciones (agrupadas)")
    ax.set_ylabel("Distancia")
    ax.axhline(y=0.7 * max(Z[:, 2]), color="red", linestyle="--", alpha=0.5)

plt.suptitle("Dendrogramas - Comparacion de Metodos de Enlace", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "22_dendrogramas.png"))
plt.close()

# ============================================================================
# 3. SELECCION DEL NUMERO OPTIMO DE CLUSTERS
# ============================================================================
print("\n")
print("3. SELECCION DEL NUMERO OPTIMO DE CLUSTERS")

# Evaluar k=2..6 con silhouette sobre los datos completos
k_range = range(2, 7)
silhouette_scores = []

# Linkage completo sobre todos los datos
Z_full = linkage(X_sc, method="ward", metric="euclidean")

for k in k_range:
    labels_k = fcluster(Z_full, t=k, criterion="maxclust")
    sil = silhouette_score(X_sc, labels_k)
    silhouette_scores.append(sil)
    print(f"  k={k}: Silhouette = {sil:.4f}")

best_k = list(k_range)[np.argmax(silhouette_scores)]
print(f"\n  k optimo = {best_k} (Silhouette = {max(silhouette_scores):.4f})")

# Grafico de silhouette por k
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    list(k_range), silhouette_scores, "o-", color="#3498db", linewidth=2, markersize=8
)
ax.set_xlabel("Numero de clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Seleccion del Numero Optimo de Clusters")
ax.set_xticks(list(k_range))
ax.grid(alpha=0.3)

for i, (k, sil) in enumerate(zip(k_range, silhouette_scores)):
    marker = " (optimo)" if k == best_k else ""
    ax.annotate(
        f"{sil:.3f}{marker}",
        (k, sil),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=9,
    )
ax.axvline(x=best_k, color="red", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "23_silhouette_por_k.png"))
plt.close()

# ============================================================================
# 4. CLUSTERING FINAL (k optimo)
# ============================================================================
print("\n")
print(f"4. CLUSTERING JERARQUICO FINAL (k={best_k})")

cluster_labels = fcluster(Z_full, t=best_k, criterion="maxclust")
# Renombrar a 0-indexed para consistencia
cluster_labels = cluster_labels - 1

df["cluster"] = cluster_labels

print("\n  Distribucion de clusters:")
for c in sorted(df["cluster"].unique()):
    n = (df["cluster"] == c).sum()
    pct = n / len(df) * 100
    print(f"    Cluster {c}: {n} observaciones ({pct:.1f}%)")

# ============================================================================
# 5. PERFILADO DE CLUSTERS
# ============================================================================
print("\n")
print("5. PERFILADO DE CLUSTERS")

# Medias y desviaciones por cluster
profile_mean = df.groupby("cluster")[FEATURES].mean()
profile_std = df.groupby("cluster")[FEATURES].std()

print("\n  Medias por cluster:")
print(profile_mean.to_string(index=True, float_format="{:.2f}".format))
print("\n  Desviaciones tipicas por cluster:")
print(profile_std.to_string(index=True, float_format="{:.2f}".format))

# Media global para referencia
global_mean = df[FEATURES].mean()
print("\n  Media global (referencia):")
for feat in FEATURES:
    print(f"    {feat}: {global_mean[feat]:.2f}")

# Grafico de perfiles (barras agrupadas)
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(FEATURES))
n_clusters = len(profile_mean)
width = 0.8 / n_clusters

for i, (cluster_id, row) in enumerate(profile_mean.iterrows()):
    bars = ax.bar(
        x + i * width,
        row.values,
        width,
        label=f"Cluster {cluster_id}",
        color=colors[i % len(colors)],
        alpha=0.85,
    )
    for bar, val in zip(bars, row.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

ax.set_xticks(x + width * (n_clusters - 1) / 2)
ax.set_xticklabels(FEATURES, fontsize=9)
ax.set_ylabel("Media")
ax.set_title("Perfil de Clusters - Medias por Variable")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "24_perfiles_clusters.png"))
plt.close()

# ============================================================================
# 6. ANALISIS DETALLADO DE LOS CLUSTERS
# ============================================================================
print("\n")
print("6. ANALISIS DETALLADO DE LOS CLUSTERS")

# Interpretacion automatica de cada cluster
for c in sorted(df["cluster"].unique()):
    mask = df["cluster"] == c
    n = mask.sum()
    means = df.loc[mask, FEATURES].mean()
    print(f"\n  Cluster {c} ({n} obs, {n / len(df) * 100:.1f}%):")
    for feat in FEATURES:
        diff = means[feat] - global_mean[feat]
        direction = "por encima" if diff > 0 else "por debajo"
        print(
            f"    {feat}: {means[feat]:.2f} ({direction} de la media global {global_mean[feat]:.2f}, diff={diff:+.2f})"
        )

# Boxplots por cluster y variable
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, feat in zip(axes, FEATURES):
    data_by_cluster = [
        df.loc[df["cluster"] == c, feat].values for c in sorted(df["cluster"].unique())
    ]
    bp = ax.boxplot(
        data_by_cluster,
        labels=[f"C{c}" for c in sorted(df["cluster"].unique())],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors[:n_clusters]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(feat, fontsize=10)
    ax.set_ylabel("Valor")
    ax.grid(axis="y", alpha=0.3)
    # Linea de media global
    ax.axhline(
        y=global_mean[feat], color="black", linestyle="--", alpha=0.5, linewidth=1
    )

plt.suptitle("Distribuciones por Cluster y Variable", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "25_boxplots_clusters.png"))
plt.close()

# ============================================================================
# 7. VISUALIZACION DE CLUSTERS EN 2D
# ============================================================================
print("\n")
print("7. VISUALIZACION DE CLUSTERS EN 2D")

# Scatter plots: tres pares de variables
pairs = [
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "sleep_hours"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (feat_x, feat_y) in zip(axes, pairs):
    for c in sorted(df["cluster"].unique()):
        mask = df["cluster"] == c
        ax.scatter(
            df.loc[mask, feat_x],
            df.loc[mask, feat_y],
            c=colors[c % len(colors)],
            label=f"Cluster {c}",
            alpha=0.35,
            s=8,
        )
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.legend(markerscale=3, fontsize=7)
    ax.grid(alpha=0.2)

plt.suptitle(f"Visualizacion 2D de Clusters (k={best_k})", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "26_scatter_clusters.png"))
plt.close()

# ============================================================================
# 8. DIAGRAMA DE SILUETA
# ============================================================================
print("\n")
print("8. DIAGRAMA DE SILUETA")

sil_final = silhouette_score(X_sc, cluster_labels)
sil_vals = silhouette_samples(X_sc, cluster_labels)
fig, ax = plt.subplots(figsize=(8, 6))

y_lower = 10
for c in sorted(np.unique(cluster_labels)):
    c_sil = sil_vals[cluster_labels == c]
    c_sil.sort()
    size = c_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        c_sil,
        alpha=0.7,
        color=colors[c % len(colors)],
        label=f"Cluster {c} (n={size})",
    )
    ax.text(-0.05, y_lower + 0.5 * size, f"{c}", fontsize=10, fontweight="bold")
    y_lower = y_upper + 10

ax.axvline(
    x=sil_final,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Media = {sil_final:.3f}",
)
ax.set_xlabel("Silhouette")
ax.set_ylabel("Observaciones (por cluster)")
ax.set_title("Diagrama de Silueta - Clustering Jerarquico")
ax.legend(loc="upper right", fontsize=8)
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "27_diagrama_silueta.png"))
plt.close()

# Silhouette medio por cluster
print(f"\n  Silhouette global: {sil_final:.4f}")
print("  Silhouette por cluster:")
for c in sorted(np.unique(cluster_labels)):
    c_sil_mean = sil_vals[cluster_labels == c].mean()
    print(f"    Cluster {c}: {c_sil_mean:.4f}")
