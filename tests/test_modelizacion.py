"""Tests para la fase de modelizacion supervisada."""

import os

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")


class TestGraficosModelizacion:
    """Verifica que todos los graficos de modelizacion existen."""

    @pytest.mark.parametrize(
        "filename",
        [
            "15_curvas_roc.png",
            "16_matrices_confusion.png",
            "17_comparativa_metricas.png",
            "18_importancia_rf.png",
            "19_learning_curves.png",
            "20_sesgo_varianza.png",
            "21_distribucion_probabilidades.png",
        ],
    )
    def test_grafico_existe(self, filename):
        path = os.path.join(IMG_DIR, filename)
        assert os.path.isfile(path), f"Falta grafico: {filename}"

    @pytest.mark.parametrize(
        "filename",
        [
            "15_curvas_roc.png",
            "16_matrices_confusion.png",
            "17_comparativa_metricas.png",
            "18_importancia_rf.png",
            "19_learning_curves.png",
            "20_sesgo_varianza.png",
            "21_distribucion_probabilidades.png",
        ],
    )
    def test_grafico_no_vacio(self, filename):
        path = os.path.join(IMG_DIR, filename)
        assert os.path.getsize(path) > 1000, f"Grafico vacio o corrupto: {filename}"


class TestDataClean:
    """Verifica que el dataset limpio sigue integro tras modelizacion."""

    def test_data_clean_existe(self):
        path = os.path.join(BASE_DIR, "data", "data_clean.csv")
        assert os.path.isfile(path)

    def test_data_clean_filas(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "data_clean.csv"))
        assert len(df) == 7500

    def test_data_clean_columnas(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "data_clean.csv"))
        expected = {"daily_screen_time_hours", "social_media_hours",
                    "sleep_hours", "addicted_label"}
        assert set(df.columns) == expected
