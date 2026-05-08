# I. Auditoría y Preprocesamiento - Análisis Exploratorio de Datos (EDA)

**Autor:** Álvaro Inclán  
**Asignatura:** Minería de Datos - Grado en Matemáticas  
**Dataset:** Smartphone Addiction Prediction Data  

---

## 1. Descripción del dataset

### 1.1 Contexto y motivación

La adicción al smartphone es un fenómeno de creciente interés en la investigación y la salud pública. La OMS ha reconocido el uso problemático de dispositivos digitales como factor de riesgo asociado a trastornos del sueño, ansiedad y deterioro del rendimiento académico. En España, el 99,5% de los jóvenes de 16 a 24 años usa el móvil a diario, con un tiempo medio de pantalla superior a 5 horas (INE, 2024).

En este contexto, **predecir y segmentar el riesgo de adicción** a partir de datos comportamentales resulta de gran utilidad para diseñar intervenciones preventivas personalizadas.

### 1.2 Fuente y estructura general

El dataset *Smartphone Addiction Prediction Data* contiene **7 500 registros** y **16 variables** que recogen patrones de uso del smartphone. Está orientado a **clasificación binaria**, donde `addicted_label` indica si un usuario es adicto (1) o no (0).

### 1.3 Justificación de la elección

- **Relevancia social:** Problema contemporáneo con implicaciones en salud mental y rendimiento académico.
- **Adecuación al trabajo:** Mezcla de variables numéricas y categóricas, target binario y tamaño moderado (7 500 filas), lo que permite aplicar todas las técnicas requeridas.
- **Complejidad controlada:** Variables temporales, demográficas y subjetivas que ofrecen un espacio de características rico para explorar relaciones no lineales.

### 1.4 Descripción de las variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `transaction_id` | `str` | ID único del registro (excluir) |
| `user_id` | `str` | ID único del usuario (excluir) |
| `daily_screen_time_hours` | `float64` | Tiempo total de pantalla diario (h) [3.0, 12.0] |
| `social_media_hours` | `float64` | Horas diarias en redes sociales [0.5, 6.0] |
| `gaming_hours` | `float64` | Horas diarias en videojuegos [0.0, 4.0] |
| `work_study_hours` | `float64` | Horas diarias de uso productivo [0.5, 6.0] |
| `weekend_screen_time` | `float64` | Tiempo de pantalla en fines de semana (h) [3.58, 14.88] |
| `age` | `int64` | Edad del usuario [18, 35] |
| `sleep_hours` | `float64` | Duración media del sueño (h) [4.5, 9.0] |
| `notifications_per_day` | `int64` | Notificaciones recibidas al día [20, 250] |
| `app_opens_per_day` | `int64` | Aperturas de apps al día [15, 180] |
| `gender` | `str` | Male, Female, Other |
| `stress_level` | `str` | Low, Medium, High |
| `academic_work_impact` | `str` | Yes, No |
| `addiction_level` | `str` | Mild, Moderate, Severe (819 NaN, 10.9%) |
| **`addicted_label`** | **`int64`** | **0 (no adicto), 1 (adicto) - Variable objetivo** |

### 1.5 Clasificación de las variables

| Rol | Variables | Cantidad |
|-----|-----------|----------|
| **Identificadores** (excluir) | `transaction_id`, `user_id` | 2 |
| **Predictoras numéricas** | `age`, `daily_screen_time_hours`, `social_media_hours`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`, `weekend_screen_time` | 9 |
| **Predictoras categóricas** | `gender`, `stress_level`, `academic_work_impact` | 3 |
| **Candidata a excluir** (*leakage*) | `addiction_level` | 1 |
| **Variable objetivo** | `addicted_label` | 1 |

---

## 2. Análisis de valores faltantes

La auditoría de completitud revela un dataset notablemente limpio. La única variable con valores ausentes es `addiction_level` con **819 NaN (10.92%)**. El resto de variables tiene 0 faltantes. Total de celdas faltantes: 819 de 120 000 (0.68%).

![Mapa de valores faltantes](img/01_mapa_faltantes.png)

!!! note "Observación sobre `addiction_level`"
    `addiction_level` tiene una relación **determinista** con `addicted_label`: Mild -> label=0, Moderate/Severe -> label=1. Es una recodificación de la variable objetivo, por lo que debe **excluirse** del modelado para evitar *data leakage*. No se requiere imputación.

---

## 3. Estadísticos descriptivos

| Variable | Media | Mediana | Desv. Típica | Mín | Máx | Asimetría | Curtosis |
|----------|-------|---------|---------------|-----|-----|-----------|----------|
| `age` | 26.57 | 27.00 | 5.20 | 18 | 35 | -0.021 | -1.226 |
| `daily_screen_time_hours` | 7.50 | 7.53 | 2.61 | 3.00 | 12.00 | -0.011 | -1.216 |
| `social_media_hours` | 3.27 | 3.27 | 1.59 | 0.50 | 6.00 | -0.010 | -1.196 |
| `gaming_hours` | 2.01 | 2.04 | 1.15 | 0.00 | 4.00 | -0.017 | -1.185 |
| `work_study_hours` | 3.24 | 3.23 | 1.60 | 0.50 | 6.00 | +0.006 | -1.216 |
| `sleep_hours` | 6.74 | 6.72 | 1.28 | 4.50 | 9.00 | +0.019 | -1.175 |
| `notifications_per_day` | 134.26 | 134.00 | 66.59 | 20 | 250 | +0.003 | -1.189 |
| `app_opens_per_day` | 97.83 | 98.00 | 48.42 | 15 | 180 | -0.010 | -1.218 |
| `weekend_screen_time` | 9.24 | 9.26 | 2.72 | 3.58 | 14.88 | -0.009 | -1.055 |

**Observaciones:** Asimetría ~= 0 y curtosis ~= -1.2 en todas las variables. Las distribuciones son simétricas y platocúrticas (más planas que la normal), lo que sugiere un dataset generado sintéticamente.

---

## 4. Estudio de distribuciones

### 4.1 Variables numéricas

![Distribuciones numéricas](img/02_distribuciones_numericas.png)

Todas las variables numéricas se aproximan a una **distribución uniforme**. El test de Shapiro-Wilk rechaza la normalidad (alfa = 0.05) en todas ellas, con p-valores del orden de 10^-30 a 10^-38. Esto implica que se deberán usar métodos **no paramétricos** o correlación robusta (Spearman) cuando corresponda.

### 4.2 Variables categóricas y variable objetivo

![Distribuciones categóricas](img/03_distribuciones_categoricas.png)

| Variable | Distribución |
|----------|-------------|
| `stress_level` | Equilibrada (~33% por categoría) |
| `gender` | Equilibrada (~33% por categoría) |
| `academic_work_impact` | Perfectamente balanceada (50/50) |
| `addiction_level` | Desbalanceada: Moderate 43%, Severe 36.4%, Mild 20.6% *(819 NaN)* |
| **`addicted_label`** | **Desbalanceada: 70.8% adictos / 29.2% no adictos** |

!!! important "Desbalance en la variable objetivo"
    Ratio 2.42:1. Se deberá considerar SMOTE, ponderación de clases o métricas robustas (F1-score, AUC-ROC).

---

## 5. Análisis de correlaciones

![Matrices de correlación](img/04_correlaciones.png)

Las correlaciones de Pearson y Spearman son prácticamente idénticas.

### 5.1 Correlaciones significativas (|r| > 0.3)

| Par de variables | r (Pearson) | Interpretación |
|-----------------|-------------|----------------|
| `daily_screen_time_hours` <-> `weekend_screen_time` | **0.964** | Redundancia casi total |
| `daily_screen_time_hours` <-> `addicted_label` | **0.577** | Correlación fuerte con el target |
| `weekend_screen_time` <-> `addicted_label` | **0.555** | Correlación fuerte con el target |
| `social_media_hours` <-> `addicted_label` | **0.414** | Correlación moderada con el target |

### 5.2 Correlaciones con la variable objetivo

![Scatter plots con target](img/05_scatter_target.png)

| Variable | r | Relevancia |
|----------|---|------------|
| `daily_screen_time_hours` | +0.577 | *** Alta |
| `weekend_screen_time` | +0.555 | *** Alta |
| `social_media_hours` | +0.414 | ** Moderada |
| `sleep_hours` | +0.036 | Baja |
| Resto de variables | < 0.01 | Insignificante |

!!! tip "Hallazgos clave"
    - Solo **3 variables** muestran correlación relevante con la adicción.
    - `daily_screen_time_hours` y `weekend_screen_time` son casi colineales (r = 0.964): una es redundante.
    - `gaming_hours`, `work_study_hours`, `notifications_per_day` y `app_opens_per_day` no tienen relación lineal con la adicción.

---

## 6. Detección de outliers

![Boxplots de outliers](img/06_outliers_boxplots.png)

El método IQR no detecta **ningún outlier** en ninguna variable numérica. Esto, combinado con las distribuciones uniformes, confirma que el dataset ha sido generado sintéticamente con rangos acotados. No se requiere tratamiento de outliers.

---

## 7. Relación entre variables categóricas y la variable objetivo

![Categóricas vs Target](img/07_categoricas_vs_target.png)

Las proporciones de adicción son **prácticamente iguales** en todas las categorías de `stress_level` (~70%), `gender` (~70%) y `academic_work_impact` (70.8% exacto en ambas). Estas variables **no aportan poder discriminante**.

La variable `addiction_level` presenta relación **100% determinista** con el target (Mild -> 0, Moderate/Severe -> 1), confirmando que debe excluirse del modelado.

---

## 8. Distribuciones por grupo de adicción

![Violines por grupo](img/08_violines_por_grupo.png)

| Variable | Media (No adicto) | Media (Adicto) | Diferencia |
|----------|-------------------|----------------|------------|
| `daily_screen_time_hours` | 5.16 | 8.47 | **+3.31 h** |
| `weekend_screen_time` | 6.90 | 10.21 | **+3.32 h** |
| `social_media_hours` | 2.25 | 3.70 | **+1.45 h** |

El resto de variables (`age`, `gaming_hours`, `work_study_hours`, `sleep_hours`, `notifications_per_day`, `app_opens_per_day`) muestran distribuciones casi idénticas entre grupos (diferencias < 1.2 unidades).

---

## 9. Pairplot de variables clave

![Pairplot](img/09_pairplot.png)

La separación entre clases se produce principalmente en los ejes de `daily_screen_time_hours` y `social_media_hours`, con solapamiento parcial entre grupos.

---

## 10. Conclusiones del EDA y plan de preprocesamiento

### 10.1 Hallazgos principales

1. **Dataset limpio:** Solo `addiction_level` tiene faltantes (10.9%). Sin outliers, duplicados ni inconsistencias.
2. **Distribuciones uniformes y sintéticas:** Curtosis ~= -1.2, distribuciones no normales.
3. **Tres variables discriminantes:** `daily_screen_time_hours` (r=0.577), `weekend_screen_time` (r=0.555), `social_media_hours` (r=0.414).
4. **Colinealidad extrema:** `daily_screen_time_hours` y `weekend_screen_time` (r=0.964).
5. **Categóricas no informativas:** `stress_level`, `gender` y `academic_work_impact` no discriminan.
6. **Desbalance moderado** en el target: 70.8% / 29.2%.

### 10.2 Plan de preprocesamiento

| Paso | Acción | Justificación |
|------|--------|---------------|
| 1 | Eliminar `transaction_id`, `user_id` | Identificadores sin valor predictivo |
| 2 | Eliminar `addiction_level` | Data leakage |
| 3 | Eliminar `weekend_screen_time` | Colinealidad extrema (r = 0.964) |
| 4 | Codificar `gender` y `academic_work_impact` | One-hot encoding |
| 5 | Codificar `stress_level` | Ordinal encoding (Low=0, Medium=1, High=2) |
| 6 | Selección de variables (Stepwise/PCA) | Reducir dimensionalidad |
| 7 | Gestionar desbalance de clases | SMOTE, ponderación o estratificación en CV |

---

## 11. Tratamiento de valores faltantes

### 11.1 Localización y diagnóstico

El **100% de las filas con NaN** en `addiction_level` pertenecen al grupo no adicto (`addicted_label = 0`). Para determinar el mecanismo de pérdida se realizaron:

- **Test Chi^2 de independencia:** Chi^2 = 2222.51, gl = 1, p ~= 0. Se rechaza H0: los NaN dependen del target. Mecanismo **MAR**.
- **Test t de Welch:** Diferencias significativas en `daily_screen_time_hours` (4.523 vs 7.865, p ~= 0) y `social_media_hours` (2.241 vs 3.400, p = 6.86 x 10^-137), explicadas porque los NaN pertenecen exclusivamente al grupo no adicto.

![Distribuciones NaN vs OK](img/10_distribucion_nan_vs_ok.png)

### 11.2 Relación determinista y decisión

La tabla cruzada confirma correspondencia **100% determinista** entre `addiction_level` y `addicted_label`:

| addiction_level | addicted_label = 0 | addicted_label = 1 |
|----------------|--------------------|--------------------|
| (NaN) | 819 | 0 |
| Mild | 1 373 | 0 |
| Moderate | 0 | 2 874 |
| Severe | 0 | 2 434 |

![addiction_level vs target](img/11_addiction_level_vs_target.png)

!!! warning "Decisión: eliminar `addiction_level`"
    1. **Imputación trivial:** Los 819 NaN son todos label=0, se imputarían como "Mild" sin aportar información.
    2. **Data leakage:** El modelo aprendería la regla determinista en lugar de patrones reales.
    3. **Confirmación estadística:** Chi^2 (p ~= 0) y tabla cruzada 1:1.
    4. **Sin pérdida de datos:** Se eliminan columnas, no filas. Se conservan las 7 500 observaciones.

---

## 12. Ingeniería de características y selección de variables

### 12.1 Preparación previa

**Variables eliminadas:** `transaction_id`, `user_id` (IDs), `addiction_level` (leakage), `weekend_screen_time` (colinealidad).

**Codificación de categóricas:**

| Variable | Codificación | Resultado |
|----------|-------------|-----------|
| `stress_level` | Ordinal | Low=0, Medium=1, High=2 |
| `academic_work_impact` | Binaria | No=0, Yes=1 |
| `gender` | One-hot (drop_first) | `gender_Male`, `gender_Other` |

Tras estas transformaciones: **12 variables** numéricas candidatas.

### 12.2 Método: Stepwise Forward con AIC

Se utiliza el criterio de información de Akaike (AIC = 2k - 2ln(L)) con regresión logística. En cada paso se añade la variable que más reduce el AIC, deteniéndose cuando ninguna mejora el criterio.

### 12.3 Resultados del Stepwise Forward

| Paso | Variable añadida | AIC |
|------|-----------------|-----|
| 0 | *(intercepto)* | 9 064.53 |
| 1 | `daily_screen_time_hours` | 6 127.82 |
| 2 | `social_media_hours` | 3 692.61 |
| 3 | `sleep_hours` | 3 687.77 |
| 4 | *ninguna mejora* -> STOP | - |

![Evolución del AIC](img/13_evolucion_aic.png)

### 12.4 Variables seleccionadas (3) y modelo final

| Variable | Coeficiente (Logit) | p-valor | Interpretación |
|----------|--------------------:|--------:|----------------|
| `daily_screen_time_hours` | +1.193 | < 0.001 | Mayor pantalla -> mayor adicción |
| `social_media_hours` | +1.461 | < 0.001 | Mayor uso RRSS -> mayor adicción |
| `sleep_hours` | +0.086 | 0.009 | Relación positiva débil pero significativa |

Las 9 variables restantes (`age`, `gaming_hours`, `work_study_hours`, `notifications_per_day`, `app_opens_per_day`, `stress_level`, `academic_work_impact`, `gender_Male`, `gender_Other`) fueron descartadas por no mejorar el AIC.

![Importancia de variables](img/12_importancia_stepwise.png)

**Métricas del modelo logístico final:** Pseudo R^2 (McFadden) = 0.594, AIC = 3 687.77, todas las variables significativas al nivel alfa = 0.01.

### 12.5 Correlaciones del dataset final

![Correlaciones finales](img/14_correlaciones_final.png)

Las tres variables seleccionadas presentan baja correlación entre sí, confirmando que aportan información complementaria.

---

## 13. Dataset final exportado

El dataset limpio se ha guardado en `data/data_clean.csv`:

| # | Columna | Rol | Tipo |
|---|---------|-----|------|
| 1 | `daily_screen_time_hours` | Predictora | float64 |
| 2 | `social_media_hours` | Predictora | float64 |
| 3 | `sleep_hours` | Predictora | float64 |
| 4 | `addicted_label` | Target | int64 |

- **Filas:** 7 500 (sin pérdidas) | **Columnas:** 4 | **Valores faltantes:** 0

---

## 14. Validación mediante tests automatizados

Se han desarrollado tres suites de tests con pytest para garantizar la reproducibilidad:

**`tests/test_eda.py`** - 23 tests en 6 clases que validan: carga y estructura del CSV, tipos de datos, valores faltantes (solo `addiction_level`), rangos numéricos y ausencia de outliers, categorías esperadas, y generación de los 9 gráficos.

**`tests/test_tratamiento.py`** - 5 tests que validan: NaN solo en `addiction_level`, todos los NaN son label=0, relación determinista Mild->0 y Moderate/Severe->1, y generación de gráficos.

**`tests/test_ingenieria.py`** - 5 tests que validan: existencia de `data_clean.csv`, columnas esperadas, ausencia de NaN, conservación de 7 500 filas, y generación de gráficos.

```bash
uv run pytest tests/ -v
```

Todos los tests pasan correctamente y actúan como guardia de regresión para futuras fases del proyecto.

---

# II. Modelización Supervisada y Contraste

## 15. Estrategia de modelización

### 15.1 Objetivo

Contrastar modelos de **tres naturalezas distintas** para predecir la adicción al smartphone (`addicted_label`), según los requisitos del enunciado:

| Categoría | Modelo | Justificación |
|-----------|--------|---------------|
| **Baseline (lineal)** | Regresión Logística | Modelo interpretable que establece la referencia |
| **Modelo flexible** | SVM con kernel RBF | Captura relaciones no lineales en el espacio de características |
| **Ensemble (agregación)** | Random Forest | Método de bagging que reduce varianza y mejora robustez |

### 15.2 Protocolo experimental

- **División Train/Test:** 80/20 estratificado (semilla 42)
- **Validación cruzada:** StratifiedKFold con 5 folds para ajuste de hiperparámetros
- **Métrica de optimización:** AUC-ROC (robusta ante desbalance)
- **Estandarización:** StandardScaler ajustado solo en train (evita data leakage)
- **Métricas de evaluación:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

### 15.3 División de datos

| Conjunto | Observaciones | % Positivos |
|----------|---------------|-------------|
| **Train** | 6 000 | 70.8% |
| **Test** | 1 500 | 70.8% |

La estratificación garantiza que ambos conjuntos mantienen la misma proporción del target.

---

## 16. Modelo 1: Regresión Logística (Baseline)

### 16.1 Justificación

La regresión logística es el baseline natural para clasificación binaria. Como modelo lineal generalizado (GLM), proporciona coeficientes interpretables y una referencia contra la que medir modelos más complejos.

### 16.2 Ajuste de hiperparámetros

| Hiperparámetro | Valores explorados | Mejor valor |
|---------------|--------------------|-------------|
| `C` (regularización) | 0.001, 0.01, 0.1, 1, 10, 100 | **0.1** |
| `penalty` | L1, L2 | **L2** |
| `solver` | saga | saga |
| `class_weight` | None, balanced | **None** |

El valor óptimo de `C = 0.1` indica que se beneficia de una regularización moderada, penalizando coeficientes excesivamente grandes.

### 16.3 Coeficientes del modelo

| Variable | Coeficiente (estandarizado) | Interpretación |
|----------|----------------------------:|----------------|
| `daily_screen_time_hours` | +2.794 | Mayor tiempo de pantalla -> mayor riesgo |
| `social_media_hours` | +2.064 | Mayor uso de RRSS -> mayor riesgo |
| `sleep_hours` | +0.108 | Efecto positivo débil pero significativo |
| *(intercepto)* | +2.278 | Sesgo base hacia la clase positiva |

### 16.4 Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.8927 |
| **Precision** | 0.9122 |
| **Recall** | 0.9388 |
| **F1-Score** | 0.9253 |
| **AUC-ROC** | 0.9544 |

> [!NOTE]
> El modelo logístico ya consigue un AUC-ROC de 0.954, lo que indica que las relaciones lineales capturan una gran parte de la estructura del problema.

---

## 17. Modelo 2: SVM con kernel RBF (Modelo Flexible)

### 17.1 Justificación

El SVM con kernel gaussiano (RBF) proyecta los datos a un espacio de alta dimensionalidad donde las clases pueden ser separadas por un hiperplano. Permite capturar **relaciones no lineales** sin asumir una forma funcional concreta.

### 17.2 Ajuste de hiperparámetros

| Hiperparámetro | Valores explorados | Mejor valor |
|---------------|--------------------|-------------|
| `C` (coste) | 0.1, 1, 10, 100 | **100** |
| `gamma` (ancho del kernel) | scale, auto, 0.01, 0.1, 1 | **1** |
| `class_weight` | None, balanced | **None** |

Un valor alto de `C = 100` y `gamma = 1` indica que el modelo explota fronteras de decisión complejas y localizadas, adaptándose finamente a la estructura de los datos.

### 17.3 Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.9307 |
| **Precision** | 0.9553 |
| **Recall** | 0.9463 |
| **F1-Score** | 0.9508 |
| **AUC-ROC** | 0.9850 |

**Vectores soporte:** 834 (422 clase 0 + 412 clase 1), representando un 13.9% de los datos de entrenamiento.

> [!TIP]
> La mejora de +3.06 puntos en AUC-ROC respecto al baseline confirma la presencia de relaciones no lineales que la regresión logística no captura.

---

## 18. Modelo 3: Random Forest (Ensemble)

### 18.1 Justificación

Random Forest es un método de agregación (*bagging*) que combina múltiples árboles de decisión entrenados sobre submuestras aleatorias. Reduce la varianza sin aumentar significativamente el sesgo, y proporciona una medida natural de importancia de variables.

### 18.2 Ajuste de hiperparámetros

| Hiperparámetro | Valores explorados | Mejor valor |
|---------------|--------------------|-------------|
| `n_estimators` | 100, 200, 500 | **100** |
| `max_depth` | 3, 5, 10, None | **None** |
| `min_samples_split` | 2, 5, 10 | **2** |
| `class_weight` | None, balanced | **None** |

La profundidad ilimitada (`max_depth=None`) y `min_samples_split=2` indican que los árboles se desarrollan completamente. El modelo con solo 100 árboles ya converge.

### 18.3 Importancia de variables (Gini)

| Variable | Importancia | Interpretación |
|----------|------------:|----------------|
| `daily_screen_time_hours` | **0.556** | Variable dominante |
| `social_media_hours` | **0.382** | Segunda en importancia |
| `sleep_hours` | 0.062 | Contribución marginal |

### 18.4 Resultados

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 0.9327 |
| **Precision** | 0.9563 |
| **Recall** | 0.9482 |
| **F1-Score** | 0.9522 |
| **AUC-ROC** | 0.9891 |

![Importancia de variables Random Forest](img/18_importancia_rf.png)

---

## 19. Comparativa de modelos

### 19.1 Tabla resumen de métricas en Test

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|:--------:|:---------:|:------:|:--------:|:-------:|
| Regresión Logística | 0.8927 | 0.9122 | 0.9388 | 0.9253 | 0.9544 |
| SVM (RBF) | 0.9307 | 0.9553 | 0.9463 | 0.9508 | 0.9850 |
| **Random Forest** | **0.9327** | **0.9563** | **0.9482** | **0.9522** | **0.9891** |

![Comparativa de métricas](img/17_comparativa_metricas.png)

### 19.2 Curvas ROC

![Curvas ROC](img/15_curvas_roc.png)

Las tres curvas muestran un excelente poder discriminante. Random Forest y SVM presentan curvas prácticamente superpuestas, ambas claramente superiores al baseline logístico.

### 19.3 Matrices de confusión

![Matrices de confusión](img/16_matrices_confusion.png)

### 19.4 Distribución de probabilidades predichas

![Distribución de probabilidades](img/21_distribucion_probabilidades.png)

Los modelos flexibles (SVM y RF) producen distribuciones más separadas entre clases, lo que indica mayor confianza en las predicciones.

---

## 20. Análisis del compromiso sesgo-varianza

### 20.1 Métricas de Train vs Validación (AUC-ROC)

| Modelo | CV Train | CV Validación | Gap | Diagnóstico |
|--------|:--------:|:-------------:|:---:|-------------|
| Regresión Logística | 0.9529 | 0.9529 | **0.0000** | Buen ajuste |
| SVM (RBF) | 0.9916 | 0.9868 | **0.0048** | Buen ajuste |
| Random Forest | 1.0000 | 0.9900 | **0.0100** | Buen ajuste |

![Sesgo-Varianza](img/20_sesgo_varianza.png)

### 20.2 Learning curves

![Learning Curves](img/19_learning_curves.png)

**Interpretación:**

- **Regresión Logística:** Las curvas de train y validación convergen rápidamente y se mantienen juntas. Gap nulo (0.000) confirma que el modelo está limitado por su **sesgo** (no puede capturar relaciones no lineales), no por varianza.
- **SVM (RBF):** Gap mínimo (0.005). El modelo tiene suficiente flexibilidad para capturar la no linealidad sin sobreajustar. Buen equilibrio.
- **Random Forest:** Gap de 0.010 con AUC train = 1.000, lo que indica que los árboles individuales memorizan el train, pero la agregación por bagging controla eficazmente la varianza. El rendimiento en validación (0.990) es el más alto.

> [!IMPORTANT]
> Ningún modelo presenta sobreajuste problemático. El gap máximo (Random Forest, 0.010) es muy pequeño, confirmando que la validación cruzada con 5 folds y el tamaño de la muestra (6 000 train) son adecuados para los tres modelos.

---

## 21. Conclusiones de la modelización

### 21.1 Ranking de modelos

1. **Random Forest** (AUC = 0.989) - Mejor rendimiento global con gap sesgo-varianza controlado.
2. **SVM (RBF)** (AUC = 0.985) - Rendimiento muy similar al RF con mejor equilibrio sesgo-varianza.
3. **Regresión Logística** (AUC = 0.954) - Baseline sólido pero limitado por su linealidad.

### 21.2 Hallazgos clave

- La **mejora del baseline a los modelos flexibles** (+3.5 puntos AUC) confirma la existencia de relaciones no lineales en los datos, aunque la mayor parte de la estructura es capturada linealmente.
- Las dos variables más importantes (`daily_screen_time_hours` y `social_media_hours`) dominan las predicciones en todos los modelos, con `sleep_hours` aportando información complementaria marginal.
- El desbalance de clases (70.8% / 29.2%) no requirió tratamiento especial (`class_weight=None` fue óptimo en los tres modelos), gracias a que el desbalance es moderado y los modelos discriminan bien.
- Los tres modelos alcanzan alta precision y recall simultáneamente, con F1 > 0.92 en todos los casos.
