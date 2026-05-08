# Ciclo de vida del dato 

> Proyecto final — Minería de datos · Grado en Matemáticas · UNIE Universidad

[![CI](https://github.com/alvaroinclan/proyecto_mineria_datos/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaroinclan/proyecto_mineria_datos/actions/workflows/ci.yml)
[![Docs](https://github.com/alvaroinclan/proyecto_mineria_datos/actions/workflows/docs.yml/badge.svg)](https://alvaroinclan.github.io/proyecto_mineria_datos/)
[![Coverage](https://codecov.io/gh/alvaroinclan/proyecto_mineria_datos/graph/badge.svg)](https://codecov.io/gh/alvaroinclan/proyecto_mineria_datos)
[![Version](https://img.shields.io/github/v/release/alvaroinclan/proyecto_mineria_datos)](https://github.com/alvaroinclan/proyecto_mineria_datos/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Description

Este proyecto aborda la predicción y segmentación del riesgo de adicción al smartphone utilizando técnicas de Minería de Datos. A través del dataset *Smartphone Addiction Prediction Data*, se lleva a cabo el ciclo de vida completo del dato: desde el preprocesamiento y la validación de la calidad, pasando por el modelado predictivo (Regresión Logística, SVM, Random Forest) y la segmentación no supervisada (Clustering Jerárquico), hasta la interpretación final del impacto de cada variable mediante análisis de residuos y permutation importance.

## Documentation

Full documentation at **[alvaroinclan.github.io/proyecto_mineria_datos](https://alvaroinclan.github.io/proyecto_mineria_datos/)**

## Installation

  ```bash
  git clone https://github.com/alvaroinclan/proyecto_mineria_datos.git
  cd proyecto_mineria_datos
  pip install uv
  uv sync --group dev
  ```

## Data Download

Data is not included in the repository. To download Kaggle

## Usage

  ```bash
  uv run pytest                          # run tests
  uv run pytest --cov=src -v             # tests with coverage
  uv run ruff check .                    # lint
  uv run ruff format .                   # format
  uv run mkdocs serve                    # preview docs at localhost:8000
  ```

## Testing y Cobertura (Coverage)

**Nota sobre Codecov (0%):** Los tests de este proyecto (`tests/`) actúan como pruebas de integración que validan el correcto resultado del pipeline (los datasets limpios y las gráficas en `docs/img/`), pero no importan directamente el código de la carpeta `src/`. Por este motivo, herramientas como `pytest-cov` o Codecov registran un 0% de cobertura, lo cual es el comportamiento esperado debido a la arquitectura orientada a scripts (pipeline) utilizada.

## Project Structure

  ```
  proyecto_mineria_datos/
  ├── .github/workflows/   # CI/CD pipelines
  ├── data/                # Data files (not committed — see .gitignore)
  ├── docs/                # MkDocs documentation sources
  ├── src/                 # Source package
  ├── tests/               # Unit and integration tests
  ├── mkdocs.yml
  ├── pyproject.toml
  └── README.md
  ```

## Author

**Álvaro Inclán** · [github.com/alvaroinclan](https://github.com/alvaroinclan)



---

*Minería de datos · 4º Grado en Matemáticas · UNIE Universidad · 2025–2026*