# Model Card — `titanic_model_v1`

## 1. Propósito y uso previsto
Modelo predictivo tabular desarrollado para estimar la probabilidad histórica de supervivencia de un pasajero del RMS Titanic (1912) a partir de variables demográficas y de viaje conocidas al embarcar, apoyando estudios retrospectivos de riesgo marítimo.

## 2. Datos de entrenamiento y trazabilidad
- **Fuente:** Dataset histórico del Titanic ingerido mediante `flows/00_ingestion.py` (`891` registros etiquetados, `418` sin etiqueta para prueba de humo).
- **Capa Silver (`data/silver/titanic_silver.parquet`):** SHA-256 `1169b3fa24d4b86ae51e4b7402d10c24d464c234da3560e691dd56bf02073a0e`.
- **División:** 80 % entrenamiento ($n = 712$) / 20 % prueba estratificada ($n = 179$), `RANDOM_STATE = 42`. El artefacto final (`models/titanic_model_v1.joblib`) se reentrenó con los 891 registros etiquetados tras congelar hiperparámetros.

## 3. Variables utilizadas y excluidas
- **Numéricas (`NUM_FEATURES`):** `Age` (imputada dentro del pipeline por media de `Sex` y `Pclass` a partir de `AgeOriginal`), `SibSp`, `Parch`, `Fare`, `Familia`.
- **Categóricas (`CAT_FEATURES`):** `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone` (codificadas mediante `OneHotEncoder` con agrupación de categorías infrecuentes).
- **Excluidas:** `PassengerId`, `Ticket`, `Name` (identificadores), `HasCabin`, `TCabin`, `Cabin` (77.1 % nulos y riesgo de sesgo de registro post-evento) y codificaciones ordinales arbitrarias (`NSex`, `NEmbarked`, `NCabin`, `NTitulo`).

## 4. Desempeño cuantitativo (Conjunto de prueba, $n = 179$, abierto una sola vez)
- **ROC-AUC:** `0.8395` (IC 95 % bootstrap: `[0.7656, 0.9030]`)
- **Exactitud (*Accuracy*):** `0.7989` (supera el baseline ingenuo `0.616` y la regla simple de género)
- **Puntaje F1 (umbral = `0.50`):** `0.7000`

## 5. Especificaciones técnicas y reproducibilidad
- **Estimador:** `HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, max_iter=200, l2_regularization=1.0, random_state=42)`
- **Entorno:** Python `3.14.4`, `scikit-learn 1.9.1`, `pandas 3.1.1`
- **Reproducción completa desde cero:**
  - `make install`
  - `make test`
  - `make pipeline`
