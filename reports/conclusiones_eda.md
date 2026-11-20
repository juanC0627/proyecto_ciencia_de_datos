# Conclusiones del Análisis Exploratorio de Datos (EDA) — Caso Titanic

## 1. Hallazgos principales

1. **Supervivencia general del pasaje ($n = 891$):** Sobrevivió el **38.4 %** ($342 / 891$) de los pasajeros etiquetados, mientras que el **61.6 %** ($549 / 891$) falleció.
2. **Fuerte disparidad por sexo ($n = 891$):** Las mujeres presentaron una tasa de supervivencia del **74.2 %** ($233 / 314$), frente al **18.9 %** ($109 / 577$) de los hombres. La prueba de independencia $\chi^2$ arroja $\chi^2 = 260.72$ ($gl = 1$, $p = 1.19 \times 10^{-58}$), confirmando que la asociación no es atribuible al azar.
3. **Gradiente socioeconómico por clase ($n = 891$):** En 1.ª clase sobrevivió el **63.0 %** ($136 / 216$), en 2.ª clase el **47.3 %** ($87 / 184$) y en 3.ª clase apenas el **24.2 %** ($119 / 491$). Al cruzar clase y sexo, las mujeres de 1.ª y 2.ª clase superaron el **92 %** de supervivencia, mientras que los hombres de 2.ª y 3.ª clase quedaron por debajo del **16 %**.
4. **Prioridad relativa de menores ($n = 83$ niños conocidos con `Age < 16`):** Los menores de 16 años alcanzaron una tasa de supervivencia del **59.0 %** ($49 / 83$), superior a la de los hombres adultos (**16.4 %**), aunque fuertemente condicionada por la clase del tiquete.
5. **Tarifas atípicas y concentración en 1.ª clase ($n = 891$):** La distribución de `Fare` es fuertemente asimétrica a la derecha ($\text{media} = 32.20$, $\text{mediana} = 14.45$, $\text{skew} = 4.79$). Mediante la regla $1.5 \cdot \text{IQR}$ se identificaron **116 valores atípicos** en tarifa (13.0 % del total), pertenecientes mayoritariamente a camarotes de primera clase y con una tasa de supervivencia sustancialmente mayor que el resto.

## 2. Decisiones de limpieza y sesgos introducidos

- **Imputación de `Age` por género (177 nulos, 19.87 %):** Rellenar con el promedio por sexo redujo la desviación estándar de `14.53` a `13.01` años y concentró artificialmente la densidad alrededor de las medias grupales. Por ello se conservó `AgeOriginal` para el análisis gráfico y el modelado.
- **Regla de `Person` ante edad desconocida (`NaN < 16` $\rightarrow$ `False`):** Los 177 pasajeros sin edad registrada quedaron clasificados según su `Sex` (`male` o `female`), subestimando la cantidad real de niños a bordo. Si entre los 177 sin edad existiera la misma proporción de menores que en los 714 con edad conocida ($83 / 714 \approx 11.62\%$), faltarían aproximadamente **21 niños** por identificar.
- **Consolidación de títulos (`Master` $\rightarrow$ `Mr`):** Agrupar `Master` dentro de `Mr` mezcla niños varones con hombres adultos en la variable `TTitulo`; este efecto queda compensado al conservar la variable `Person` y la edad numérica.
- **Categoría explícita `'X'` en `TCabin` (687 nulos, 77.10 %):** Asignar `'X'` evita perder tres cuartas partes de la muestra, aunque agrupa tanto a pasajeros sin camarote asignado como a registros administrativos incompletos.

## 3. Variables de confusión detectadas

- **Puerto de embarque (`Embarked`) vs. Clase (`Pclass`):** A nivel agregado, quienes embarcaron en Cherbourg (`C`) muestran mayor supervivencia (~55.4 %) que quienes embarcaron en Southampton (`S`, ~33.7 %). Al estratificar por `Pclass`, se observa que más de la mitad de los pasajeros de Cherbourg viajaban en 1.ª clase, mientras que Southampton concentraba la mayoría de 3.ª clase; gran parte del efecto aparente del puerto responde a la composición socioeconómica del pasaje.

## 4. Qué NO podemos afirmar (límites del análisis)

- **Asociación no es causalidad:** Los patrones observados reflejan correlaciones históricas de un evento único (el hundimiento del RMS Titanic en abril de 1912) y protocolos específicos de evacuación (*"mujeres y niños primero"*), no leyes físicas universales.
- **Datos incompletos y sesgo de supervivencia en registros:** El 77.1 % de las cabinas y el 19.9 % de las edades faltan; además, los registros de cabina y edad pudieron reconstruirse con mayor facilidad entre quienes sobrevivieron para declarar.
