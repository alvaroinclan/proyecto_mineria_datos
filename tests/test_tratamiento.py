"""
Tests para validar el tratamiento de valores faltantes.


"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data base.csv")
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")

DF = pd.read_csv(DATA_PATH)


def test_nan_solo_en_addiction_level():
    """Solo addiction_level tiene valores faltantes."""
    cols_con_nan = DF.columns[DF.isnull().any()].tolist()
    assert cols_con_nan == ["addiction_level"]


def test_todos_nan_son_label_0():
    """El 100% de filas con NaN en addiction_level tienen addicted_label = 0."""
    nan_labels = DF.loc[DF["addiction_level"].isnull(), "addicted_label"]
    assert (nan_labels == 0).all()


def test_relacion_determinista_mild():
    """Todos los Mild tienen addicted_label = 0."""
    mild = DF[DF["addiction_level"] == "Mild"]
    assert (mild["addicted_label"] == 0).all()


def test_relacion_determinista_moderate_severe():
    """Todos los Moderate y Severe tienen addicted_label = 1."""
    mod_sev = DF[DF["addiction_level"].isin(["Moderate", "Severe"])]
    assert (mod_sev["addicted_label"] == 1).all()


def test_graficos_tratamiento_existen():
    """Los gráficos del tratamiento fueron generados."""
    for img in ["10_distribucion_nan_vs_ok.png", "11_addiction_level_vs_target.png"]:
        assert os.path.isfile(os.path.join(IMG_DIR, img)), f"Falta: {img}"
