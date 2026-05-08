"""Tests para la fase de clustering"""

import os

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "docs", "img")


class TestGraficosClustering:
    """Verifica que todos los graficos de clustering existen y no estan vacios."""

    @pytest.mark.parametrize(
        "filename",
        [
            "22_dendrogramas.png",
            "23_silhouette_por_k.png",
            "24_perfiles_clusters.png",
            "25_clusters_vs_etiquetas.png",
            "26_scatter_clusters_vs_real.png",
            "27_diagrama_silueta.png",
        ],
    )
    def test_grafico_existe(self, filename):
        path = os.path.join(IMG_DIR, filename)
        assert os.path.isfile(path), f"Falta grafico: {filename}"

    @pytest.mark.parametrize(
        "filename",
        [
            "22_dendrogramas.png",
            "23_silhouette_por_k.png",
            "24_perfiles_clusters.png",
            "25_boxplots_clusters.png",
            "26_scatter_clusters.png",
            "27_diagrama_silueta.png",
        ],
    )
    def test_grafico_no_vacio(self, filename):
        path = os.path.join(IMG_DIR, filename)
        assert os.path.getsize(path) > 1000, f"Grafico corrupto: {filename}"
