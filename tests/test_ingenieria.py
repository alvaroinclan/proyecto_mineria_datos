"""
Tests para variable selection e ingeniería de variables.

"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "data_clean.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")

EXPECTED_COLS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "sleep_hours",
    "addicted_label",
]


def test_data_clean_existe():
    """The clean CSV was generated."""
    assert os.path.isfile(CLEAN_PATH), f"Missing: {CLEAN_PATH}"


def test_columnas_esperadas():
    """The clean CSV has exactly the selected columns + target."""
    df = pd.read_csv(CLEAN_PATH)
    assert list(df.columns) == EXPECTED_COLS


def test_sin_valores_faltantes():
    """The clean dataset has no NaN values."""
    df = pd.read_csv(CLEAN_PATH)
    assert df.isnull().sum().sum() == 0


def test_filas_conservadas():
    """All 7500 original rows are preserved."""
    df = pd.read_csv(CLEAN_PATH)
    assert len(df) == 7500


def test_graficos_ingenieria():
    """The 3 charts from ingenieria_variables.py were generated."""
    for img in [
        "12_importancia_stepwise.png",
        "13_evolucion_aic.png",
        "14_correlaciones_final.png",
    ]:
        assert os.path.isfile(os.path.join(IMG_DIR, img)), f"Missing: {img}"
