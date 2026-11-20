# Formulación del Problema de Machine Learning y Selección de Modelo

## 1. Definición del problema y métrica

| Decisión | Justificación |
|---|---|
| **Tipo de problema** | Clasificación binaria supervisada (`Survived` $\in \{0, 1\}$). |
| **Variable objetivo** | `Survived` (1 = sobrevivió al naufragio, 0 = no sobrevivió). |
| **Unidad de análisis y momento de predicción** | Un pasajero individual, evaluado exclusivamente con información disponible **al momento del embarque**. |
| **Métrica principal** | **ROC-AUC**, porque evalúa la capacidad de discriminación probabilística en todos los umbrales posibles sin verse distorsionada por el desbalance de clases (38.4 % positivos). |
| **Métricas secundarias** | Exactitud (*accuracy*), precisión, *recall* y puntaje F1 al umbral operativo (`0.50`). |
| **Costo de errores** | Para una aseguradora que modela riesgo histórico y protocolos de evacuación, un falso negativo (predecir fallecimiento cuando sobrevive) subestima resiliencia, mientras un falso positivo (predecir supervivencia cuando fallece) subestima la exposición al riesgo actuarialmente crítico. |
| **Criterio de éxito** | Superar holgadamente el baseline ingenuo (`0.616` de exactitud) y superar la regla determinista de género (`~0.787` de exactitud en entrenamiento) manteniendo estabilidad entre pliegues. |

## 2. Selección de modelos candidatos (Validación Cruzada Estratificada, $k = 5$)

Se evaluaron seis candidatos mediante validación cruzada estratificada sobre `X_train` ($n = 712$), integrando `GroupAgeImputer` y `ColumnTransformer` dentro del `Pipeline` para prevenir cualquier fuga de datos:

- **Finalista 1 — `HistGradientBoostingClassifier` (`hgb`):** Mejor equilibrio entre discriminación (`ROC-AUC` superior a 0.86 en CV), manejo nativo de interacciones no lineales entre clase, sexo, tarifa y edad, y baja varianza tras regularización (`learning_rate=0.03`, `max_depth=3`, `l2_regularization=1.0`).
- **Finalista 2 — `RandomForestClassifier` (`forest`):** Desempeño competitivo en CV, pero mayor brecha entre entrenamiento y validación (mayor tendencia al sobreajuste en muestra pequeña).
- **Finalista 3 — `LogisticRegression` (`logreg`):** Modelo lineal altamente interpretable mediante *odds ratios*, ideal como referencia paramétrica auditada.

**Decisión de ablación sobre cabina (`USE_CABIN_FEATURES = False`):** Aunque incluir `HasCabin` y `TCabin` aporta una leve variación marginal en CV, se excluyen del modelo final porque presentan 77.1 % de valores faltantes y un alto riesgo de ser un **artefacto del registro posterior al desastre** (mayor facilidad para documentar la cabina de los sobrevivientes).

## 3. Limitaciones y consideraciones éticas

- **Tamaño muestral e incertidumbre:** El conjunto de prueba contiene $n = 179$ pasajeros, lo que produce intervalos de confianza bootstrap relativamente amplios (`ROC-AUC = 0.840`, IC 95 %: `[0.766, 0.903]`).
- **Uso responsable:** El modelo captura patrones históricos de desigualdad por género y clase social de 1912. Sirve exclusivamente para análisis actuariales retrospectivos y auditoría de protocolos históricos; **no debe emplearse** para asignar prioridades de rescate ni discriminar cobertura o tarifas por sexo o nivel socioeconómico en contextos modernos.
