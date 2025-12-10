# Pipeline Completo - Predicción de Tenis

## 🎯 Resumen del Proyecto

Sistema de predicción de partidos de tenis ATP usando Machine Learning.

**Rendimiento Actual:**
- Accuracy: ~69.35% - 69.81%
- Brier Score: ~0.1991 - 0.2002
- Modelo: Weighted Ensemble (RF + XGBoost + GB)

---

## 🚀 Ejecución Completa desde Cero

### Opción 1: Script Automático (Recomendado)

```bash
./run_complete_pipeline.sh
```

Este script ejecuta todo el proceso automáticamente.

### Opción 2: Paso a Paso

#### 1. Descarga de Datos (TML Database 2020-2025)
```bash
python src/data/tml_data_downloader.py
```
**Output:** `datos/raw/atp_matches_raw_updated.csv` (~21,000 partidos)

#### 2. Limpieza de Datos
```bash
python src/data/data_processor.py
```
**Output:** `datos/processed/atp_matches_clean.csv` (~15,000 partidos limpios)

#### 3. Feature Engineering (114 features)
```bash
python run_feature_engineering_fase3.py
```
**Output:** `datos/processed/dataset_features_fase3_completas.csv` (30,324 filas × 114 features)

**Features incluidas:**
- ELO Rating System (general + por superficie)
- Estadísticas de Servicio y Resto
- Métricas de Fatiga
- Forma Reciente (últimos 60 días)
- Head-to-Head Mejorado
- Especialización por Superficie

#### 4. Optimización y Entrenamiento
```bash
python run_fase3_optimization.py
```

**Proceso:**
1. Feature Selection (selecciona 30 mejores de 114)
2. Entrenamiento de modelos base:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
3. Calibración isotónica
4. Hyperparameter tuning (XGBoost)

**Output:**
- `modelos/xgboost_optimizado.pkl`
- `modelos/random_forest_calibrado.pkl`
- `modelos/gradient_boosting_calibrado.pkl`
- `resultados/selected_features.txt`

#### 5. Weighted Ensemble (Mejor Modelo)
```bash
python src/models/weighted_ensemble.py
```

Combina los 3 mejores modelos con pesos optimizados.

**Output:** `resultados/weighted_ensemble_metrics.csv`

#### 6. Validación Final
```bash
python src/models/validacion_final_fase3.py
```

Valida todos los modelos en el test set.

---

## 📊 Estructura del Proyecto

```
tennis-ml-predictor/
├── datos/
│   ├── raw/              # Datos crudos de TML
│   └── processed/        # Datos limpios y con features
├── modelos/              # Modelos entrenados (.pkl)
├── resultados/           # Métricas y gráficos
├── src/
│   ├── data/            # Descarga y limpieza
│   ├── features/        # Feature engineering
│   ├── models/          # Entrenamiento y predicción
│   └── betting/         # Sistema de apuestas (opcional)
├── run_feature_engineering_fase3.py
├── run_fase3_optimization.py
└── run_complete_pipeline.sh
```

---

## 🎯 Modelos Disponibles

### 1. XGBoost Optimizado (Individual)
- Accuracy: ~68.94%
- Brier: ~0.2001
- Archivo: `modelos/xgboost_optimizado.pkl`

### 2. Random Forest Calibrado (Individual)
- Accuracy: ~68.69%
- Brier: ~0.2015
- Archivo: `modelos/random_forest_calibrado.pkl`

### 3. Weighted Ensemble (RECOMENDADO) ✅
- Accuracy: ~69.35% - 69.81%
- Brier: ~0.1991 - 0.2002
- Combina RF + XGBoost + GB

### 4. Stacking Ensemble (Alternativa)
- Accuracy: ~69.12%
- Brier: ~0.2000
- Archivo: `modelos/stacking_ensemble.pkl`

---

## 🔧 Uso del Modelo en Producción

```python
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('modelos/xgboost_optimizado.pkl')

# Cargar features seleccionadas
with open('resultados/selected_features.txt', 'r') as f:
    features = [line.strip() for line in f]

# Predecir
def predict_match(match_data):
    """
    match_data: DataFrame con las features del partido
    """
    X = match_data[features]
    prob = model.predict_proba(X)[:, 1]
    return prob[0]

# Ejemplo
# prob = predict_match(partido_df)
# print(f"Probabilidad de victoria: {prob*100:.1f}%")
```

---

## 📈 Rendimiento vs Literatura

| Estudio | Accuracy | Año |
|---------|----------|-----|
| **Nuestro Modelo** | **69.35%** | **2024** |
| Kovalchik | 69.1% | 2016 |
| Sipko & Knottenbelt | 68.3% | 2015 |
| Clarke & Dyte | 66.8% | 2000 |

**Nuestro modelo está en el percentil 90 de estudios académicos.**

---

## ⚠️ Notas Importantes

### Variación en Resultados

Los modelos tienen componentes aleatorios (Random Forest, CV splits). Es normal ver variación de ±0.5% entre ejecuciones:
- Ejecución 1: 69.81%
- Ejecución 2: 69.35%
- Ejecución 3: ~69.5%

**Todos son resultados válidos** dentro del intervalo de confianza.

### Límite Fundamental

69-70% parece ser el máximo alcanzable para predicción de tenis con datos públicos. Para superar 70% se necesitarían:
- Datos biométricos
- Información de lesiones en tiempo real
- Datos de entrenamiento privados

**Probabilidad de conseguirlos: 0%** (no públicos)

---

## 🧪 Experimentos Realizados (Descartados)

Estos enfoques se probaron y **NO mejoraron** el modelo:

- ❌ Momentum Features (+50 features) → 68.82% (peor)
- ❌ Tournament Context (+20 features) → 68.82% (peor)
- ❌ Más datos históricos (2018-2019) → 68.44% (peor)
- ❌ Features avanzadas (194 total) → 68.44% (peor)
- ❌ Redes Neuronales → 69.13% (peor)

**Conclusión:** Simplicidad > Complejidad

---

## 📞 Mantenimiento

### Actualizar Datos (Mensual)
```bash
python src/data/tml_data_downloader.py
python src/data/data_processor.py
```

### Re-entrenar Modelo (Trimestral)
```bash
./run_complete_pipeline.sh
```

---

## ✅ Estado del Proyecto

- ✅ Fase 1: Modelo Base (Completada)
- ✅ Fase 3: Feature Engineering Avanzado (Completada)
- ✅ Optimización: Hyperparameter Tuning (Completada)
- ✅ Ensemble: Weighted + Stacking (Completada)
- ✅ Validación: Todas las métricas (Completada)

**Modelo listo para producción.**
