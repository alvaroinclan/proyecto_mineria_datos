"""Tests para la fase de interpretacion y conclusiones."""

import os

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")


class TestGraficosInterpretacion:
    """Verifica que todos los graficos de interpretacion existen."""

    @pytest.mark.parametrize(
        "filename",
        [
            "28_residuos_logistica.png",
            "29_hosmer_lemeshow.png",
            "30_permutation_importance.png",
            "31_gini_vs_permutation.png",
            "32_odds_ratios.png",
        ],
    )
    def test_grafico_existe(self, filename):
        path = os.path.join(IMG_DIR, filename)
        assert os.path.isfile(path), f"Falta grafico: {filename}"

    @pytest.mark.parametrize(
        "filename",
        [
            "28_residuos_logistica.png",
            "29_hosmer_lemeshow.png",
            "30_permutation_importance.png",
            "31_gini_vs_permutation.png",
            "32_odds_ratios.png",
        ],
    )
    def test_grafico_no_vacio(self, filename):
        path = os.path.join(IMG_DIR, filename)
        assert os.path.getsize(path) > 1000, f"Grafico corrupto: {filename}"
