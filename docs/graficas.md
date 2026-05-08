# Galería de Gráficas

Todos los gráficos generados durante el preprocesamiento del dataset.

---

## EDA — Análisis Exploratorio de Datos

### 1. Mapa de valores faltantes

![Mapa de valores faltantes](img/01_mapa_faltantes.png)

---

### 2. Distribuciones — Variables numéricas

![Distribuciones numéricas](img/02_distribuciones_numericas.png)

---

### 3. Distribuciones — Variables categóricas

![Distribuciones categóricas](img/03_distribuciones_categoricas.png)

---

### 4. Matrices de correlación (Pearson y Spearman)

![Correlaciones](img/04_correlaciones.png)

---

### 5. Scatter plots — Top correlaciones con `addicted_label`

![Scatter plots](img/05_scatter_target.png)

---

### 6. Detección de outliers (Boxplots)

![Outliers](img/06_outliers_boxplots.png)

---

### 7. Variables categóricas vs variable objetivo

![Categóricas vs target](img/07_categoricas_vs_target.png)

---

### 8. Violin plots — Distribuciones por grupo de adicción

![Violin plots](img/08_violines_por_grupo.png)

---

### 9. Pairplot — Variables clave por grupo

![Pairplot](img/09_pairplot.png)

---

## Tratamiento de Valores Faltantes

### 10. Distribuciones: filas con NaN vs sin NaN

Comparación de las distribuciones de las variables numéricas principales entre las filas donde `addiction_level` es NaN y donde no lo es. Las diferencias en `daily_screen_time_hours` y `social_media_hours` se explican por la asociación de los NaN con el grupo no adicto.

![Distribuciones NaN vs OK](img/10_distribucion_nan_vs_ok.png)

---

### 11. Tabla cruzada: addiction_level × addicted_label

Visualización de la relación determinista entre `addiction_level` y `addicted_label`. Los NaN corresponden exclusivamente a `addicted_label = 0`.

![addiction_level vs target](img/11_addiction_level_vs_target.png)

---

## Ingeniería de Características y Selección de Variables

### 12. Importancia de variables seleccionadas (Stepwise Forward)

Coeficientes absolutos del modelo logístico final. Las variables en verde son estadísticamente significativas (p < 0.05).

![Importancia stepwise](img/12_importancia_stepwise.png)

---

### 13. Evolución del AIC — Stepwise Forward

Reducción progresiva del AIC al incorporar cada variable. La mayor ganancia se produce con `daily_screen_time_hours` y `social_media_hours`.

![Evolución AIC](img/13_evolucion_aic.png)

---

### 14. Correlaciones — Variables seleccionadas

Matriz de correlación del dataset final con las 3 variables seleccionadas y el target. La baja correlación entre predictoras confirma que no hay redundancia.

![Correlaciones finales](img/14_correlaciones_final.png)
