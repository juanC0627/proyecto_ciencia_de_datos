# Taller Titanic — Buenas Prácticas en Ciencia de Datos e IA

Proyecto reproducible y modular siguiendo la arquitectura **Medallion** (`Bronze -> Silver -> Gold -> ML`) para el análisis exploratorio y modelado predictivo de supervivencia en el RMS Titanic.

## Estructura del repositorio

- `flows/00_ingestion.py`: Descarga los datos crudos, separa registros etiquetados (`891`) y sin etiqueta (`418`) en `data/bronze/`, calcula huellas SHA-256 y genera `_quality_report.json`.
- `flows/01_data_engineering.py`: Transformaciones puras de limpieza, validación de contratos e ingeniería de características (`Person`, `TCabin`, `HasCabin`, `Familia`, `Alone`, `TTitulo`) hacia `data/silver/`.
- `flows/03_aggregation.py`: Tablas analíticas agregadas y prueba $\chi^2$ en `data/gold/`.
- `flows/05_ml_models.py`: Entrenamiento sin fuga de datos (`GroupAgeImputer` + `ColumnTransformer` dentro de `Pipeline`), evaluación única en prueba y serialización con metadatos en `models/`.
- `notebooks/`: `02_exploracion.ipynb` (EDA) y `04_modelado.ipynb` (validación cruzada, calibración, interpretabilidad, ablación y prueba de humo).
- `reports/`: `conclusiones_eda.md`, `problema_ml.md` y `model_card.md`.
- `tests/`: Suite de 29 pruebas unitarias con `pytest`.

## Ejecución rápida

```bash
python -m venv .venv && source .venv/bin/activate
cp .env.example .env
make install
make test
make lint
make pipeline
