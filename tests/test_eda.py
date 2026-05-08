"""
Tests para validar el EDA del dataset de adicción al smartphone.

"""

import os
import pandas as pd
import numpy as np

# Ruta al CSV
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data base.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")

# Columnas esperadas según la descripción del dataset
EXPECTED_COLUMNS = [
    "transaction_id", "user_id", "age", "gender",
    "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time", "stress_level",
    "academic_work_impact", "addiction_level", "addicted_label",
]

NUM_COLS = [
    "age", "daily_screen_time_hours", "social_media_hours",
    "gaming_hours", "work_study_hours", "sleep_hours",
    "notifications_per_day", "app_opens_per_day", "weekend_screen_time",
]


# Carga del dataset
def _load_df():
    """Carga el dataframe una sola vez para reutilizarlo en los tests."""
    return pd.read_csv(DATA_PATH)


DF = _load_df()

## Usamos == como separador entre secciones para organizar el código que si no me lío

# ============================================================================
# 1. Tests de carga y estructura
# ============================================================================
class TestCargaYEstructura:
    """Valida que el CSV se carga correctamente y tiene la forma esperada."""

    def test_csv_existe(self):
        assert os.path.isfile(DATA_PATH), f"No se encontró el archivo {DATA_PATH}"

    def test_num_filas(self):
        assert len(DF) == 7500, f"Se esperaban 7500 filas, se encontraron {len(DF)}"

    def test_num_columnas(self):
        assert len(DF.columns) == 16, (
            f"Se esperaban 16 columnas, se encontraron {len(DF.columns)}"
        )

    def test_nombres_columnas(self):
        assert list(DF.columns) == EXPECTED_COLUMNS, (
            f"Columnas inesperadas: {list(DF.columns)}"
        )


# ============================================================================
# 2. Tests de tipos de datos
# ============================================================================
class TestTiposDeDatos:
    """Verifica que cada columna tiene el dtype correcto."""

    def test_age_es_entero(self):
        assert DF["age"].dtype in (np.int64, np.int32)

    def test_screen_time_es_float(self):
        assert DF["daily_screen_time_hours"].dtype == np.float64

    def test_addicted_label_es_entero(self):
        assert DF["addicted_label"].dtype in (np.int64, np.int32)

    def test_gender_es_string(self):
        assert DF["gender"].dtype == object or pd.api.types.is_string_dtype(DF["gender"])


# ============================================================================
# 3. Tests de valores faltantes
# ============================================================================
class TestValoresFaltantes:
    """Verifica los valores faltantes detectados en el EDA."""

    def test_unica_variable_con_nan(self):
        cols_con_nan = DF.columns[DF.isnull().any()].tolist()
        assert cols_con_nan == ["addiction_level"], (
            f"Se esperaba solo 'addiction_level' con NaN, se encontraron: {cols_con_nan}"
        )

    def test_cantidad_nan_addiction_level(self):
        n_nan = DF["addiction_level"].isnull().sum()
        assert n_nan == 819, f"Se esperaban 819 NaN, se encontraron {n_nan}"

    def test_sin_duplicados_transaction_id(self):
        n_dup = DF["transaction_id"].duplicated().sum()
        assert n_dup == 0, f"Se encontraron {n_dup} transaction_id duplicados"


# ============================================================================
# 4. Tests de rangos numéricos
# ============================================================================
class TestRangosNumericos:
    """Verifica que las variables numéricas están en rangos razonables."""

    def test_age_rango(self):
        assert DF["age"].min() >= 0 and DF["age"].max() <= 120

    def test_screen_time_no_negativo(self):
        assert (DF["daily_screen_time_hours"] >= 0).all()

    def test_sleep_hours_rango(self):
        assert DF["sleep_hours"].min() >= 0 and DF["sleep_hours"].max() <= 24

    def test_target_binario(self):
        assert set(DF["addicted_label"].unique()) == {0, 1}

    def test_sin_outliers_iqr(self):
        """Confirma que no hay outliers por IQR en ninguna numérica."""
        for col in NUM_COLS:
            s = DF[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
            assert n_out == 0, f"{col} tiene {n_out} outliers IQR"


# ============================================================================
# 5. Tests de variables categóricas
# ============================================================================
class TestCategoricas:
    """Valida las categorías de las variables categóricas."""

    def test_gender_categorias(self):
        assert set(DF["gender"].unique()) == {"Male", "Female", "Other"}

    def test_stress_level_categorias(self):
        assert set(DF["stress_level"].unique()) == {"Low", "Medium", "High"}

    def test_academic_work_impact_categorias(self):
        assert set(DF["academic_work_impact"].unique()) == {"Yes", "No"}

    def test_addiction_level_categorias(self):
        cats = set(DF["addiction_level"].dropna().unique())
        assert cats == {"Mild", "Moderate", "Severe"}


# ============================================================================
# 6. Tests de gráficos generados por el EDA
# ============================================================================
class TestGraficosGenerados:
    """Verifica que el script eda.py generó los gráficos esperados."""

    EXPECTED_IMAGES = [
        "01_mapa_faltantes.png",
        "02_distribuciones_numericas.png",
        "03_distribuciones_categoricas.png",
        "04_correlaciones.png",
        "05_scatter_target.png",
        "06_outliers_boxplots.png",
        "07_categoricas_vs_target.png",
        "08_violines_por_grupo.png",
        "09_pairplot.png",
    ]

    def test_directorio_img_existe(self):
        assert os.path.isdir(IMG_DIR), f"No existe el directorio {IMG_DIR}"

    def test_graficos_existen(self):
        for img in self.EXPECTED_IMAGES:
            path = os.path.join(IMG_DIR, img)
            assert os.path.isfile(path), f"Falta el gráfico: {img}"

    def test_graficos_no_vacios(self):
        for img in self.EXPECTED_IMAGES:
            path = os.path.join(IMG_DIR, img)
            if os.path.isfile(path):
                assert os.path.getsize(path) > 1000, (
                    f"El gráfico {img} parece vacío ({os.path.getsize(path)} bytes)"
                )
