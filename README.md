# 🎾 Tennis ML Predictor

Sistema de predicción de partidos de tenis usando Machine Learning con probabilidades calibradas para apuestas deportivas.

## 📊 Resultados Actuales (Fase 2 Completada)

- **Accuracy**: 69.82% en test set 2025
- **Brier Score**: 0.1991 (calibración excelente)
- **ECE**: 0.0222 (calibración casi perfecta)
- **ROI en Backtesting**: 57.41% (excepcional)
- **Modelo**: Random Forest con 30 features seleccionadas

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8+
- pip
- Git

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/tennis-ml-predictor.git
cd tennis-ml-predictor

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear estructura de carpetas
mkdir -p datos/raw datos/processed datos/tml_database
mkdir -p modelos resultados/calibracion resultados/backtesting logs
```

---

## 📋 Pipeline Completo - Paso a Paso

### **Paso 1: Descargar Datos** 📥

Descarga datos históricos de partidos de tenis desde TML Database (2022-2025):

```bash
python src/data/tml_data_downloader.py
```

**Salida esperada**:
- `datos/tml_database/tml_tennis.db` - Base de datos SQLite con ~25,000 partidos
- Tiempo estimado: 2-3 minutos

---

### **Paso 2: Procesar Datos** 🔄

Procesa los datos raw y crea el dataset base:

```bash
python src/data/data_processor.py
```

**Salida esperada**:
- `datos/processed/dataset_base.csv` - Dataset con features básicas
- Tiempo estimado: 1 minuto

---

### **Paso 3: Feature Engineering** 🛠️

Genera las 149 features avanzadas (ELO, servicio, fatiga, forma reciente, etc.):

```bash
python run_feature_engineering_fase3.py
```

**Salida esperada**:
- `datos/processed/dataset_features_fase3_completas.csv` - Dataset con 149 features
- Tiempo estimado: 3-5 minutos

**Features generadas**:
- Sistema ELO (general y por superficie)
- Estadísticas de servicio y resto
- Métricas de fatiga
- Forma reciente (30, 60, 90 días)
- Head-to-Head histórico
- Especialización por superficie

---

### **Paso 4: Entrenamiento y Optimización** 🤖

Entrena múltiples modelos, selecciona features y optimiza hiperparámetros:

```bash
python run_fase3_optimization.py
```

**Salida esperada**:
- `modelos/random_forest_calibrado.pkl` - Modelo Random Forest calibrado
- `modelos/xgboost_calibrado.pkl` - Modelo XGBoost calibrado
- `modelos/gradient_boosting_calibrado.pkl` - Modelo Gradient Boosting calibrado
- `modelos/logistic_regression_calibrado.pkl` - Modelo Logistic Regression calibrado
- `resultados/selected_features.txt` - 30 features seleccionadas
- `resultados/model_comparison.png` - Comparación de modelos
- `resultados/hyperparameter_tuning_results.csv` - Resultados de tuning
- Tiempo estimado: 10-15 minutos

**Proceso incluye**:
1. Entrenamiento de 4 modelos base
2. Selección de 30 mejores features
3. Re-entrenamiento con features seleccionadas
4. Calibración con Isotonic Regression
5. Comparación y selección del mejor modelo

---

### **Paso 5: Validación de Calibración** 🎯

Valida que las probabilidades del modelo sean confiables:

```bash
python validacion_calibracion.py
```

**Salida esperada**:
- `resultados/calibracion/calibration_metrics.csv` - Métricas de calibración
- `resultados/calibracion/calibration_comparison_all_models.png` - Comparación visual
- `resultados/calibracion/reliability_diagrams/` - Diagramas individuales por modelo
- Tiempo estimado: 2 minutos

**Métricas validadas**:
- Brier Score < 0.20 ✅
- ECE < 0.05 ✅
- Reliability diagrams ✅

---

### **Paso 6: Backtesting** 🎲

Simula apuestas en datos históricos para validar rentabilidad:

```bash
python backtesting_fase2.py
```

**Salida esperada**:
- `resultados/backtesting/ev_threshold_comparison.csv` - Comparación de umbrales
- `resultados/backtesting/cumulative_profit_ev*.png` - Curvas de ganancias
- `resultados/backtesting/all_bets_detailed.csv` - Detalle de todas las apuestas
- Tiempo estimado: 5 minutos

**Análisis incluye**:
- 4 umbrales de EV (0%, 3%, 5%, 8%)
- ROI, Win Rate, Profit Factor
- Análisis de drawdown
- Análisis por superficie y rangos de EV

---

### **Paso 7: Reporte Final** 📊

Genera reporte HTML interactivo con todos los resultados:

```bash
python generar_reporte_fase2.py
```

**Salida esperada**:
- `resultados/REPORTE_FASE_2.html` - Reporte interactivo completo
- Tiempo estimado: 30 segundos

---

## ⚡ Pipeline Completo Automatizado

Si quieres ejecutar todo el proceso de una vez:

```bash
# Ejecuta Fase 2 completa (validación + backtesting + reporte)
python run_fase2_completa.py
```

**Nota**: Asegúrate de haber ejecutado los pasos 1-4 primero.

---

## 📁 Estructura del Proyecto

```
tennis-ml-predictor/
├── datos/
│   ├── raw/                    # Datos crudos (ignorados en Git)
│   ├── processed/              # Datasets procesados (ignorados en Git)
│   └── tml_database/           # Base de datos TML (ignorada en Git)
│
├── src/
│   ├── data/
│   │   ├── tml_data_downloader.py    # Descarga datos de TML
│   │   └── data_processor.py         # Procesa datos raw
│   │
│   ├── features/
│   │   ├── feature_engineer_completo.py  # Feature engineering completo
│   │   ├── elo_rating_system.py          # Sistema ELO
│   │   ├── features_servicio_resto.py    # Stats servicio/resto
│   │   ├── features_fatiga.py            # Métricas de fatiga
│   │   ├── features_forma_reciente.py    # Forma reciente
│   │   ├── features_h2h_mejorado.py      # Head-to-Head
│   │   └── features_superficie.py        # Especialización superficie
│   │
│   ├── models/
│   │   ├── comparacion_modelos.py        # Comparación de modelos
│   │   ├── feature_selection.py          # Selección de features
│   │   ├── hyperparameter_tuning.py      # Optimización hiperparámetros
│   │   └── weighted_ensemble.py          # Ensemble de modelos
│   │
│   └── utils/
│       └── __init__.py
│
├── modelos/                    # Modelos entrenados (ignorados en Git)
│   ├── random_forest_calibrado.pkl
│   ├── xgboost_calibrado.pkl
│   ├── gradient_boosting_calibrado.pkl
│   └── logistic_regression_calibrado.pkl
│
├── resultados/                 # Resultados y gráficos (ignorados en Git)
│   ├── calibracion/
│   ├── backtesting/
│   └── REPORTE_FASE_2.html
│
├── guiasProyecto/              # Guías de desarrollo
│   ├── FASE_2_CALIBRACION.md
│   ├── FASE_3_OPTIMIZACION.md
│   └── FASE_4_TRACKING.md
│
├── logs/                       # Logs de ejecución (ignorados en Git)
│
├── run_feature_engineering_fase3.py  # Script feature engineering
├── run_fase3_optimization.py         # Script optimización
├── validacion_calibracion.py         # Script validación
├── backtesting_fase2.py              # Script backtesting
├── generar_reporte_fase2.py          # Script reporte
├── run_fase2_completa.py             # Script pipeline completo
├── predictor_calibrado.py            # Clase predictor
│
├── requirements.txt            # Dependencias Python
├── .gitignore                  # Archivos ignorados por Git
├── README.md                   # Este archivo
└── FASE_2_RESULTADOS.md        # Documentación de resultados
```

---

## 🎯 Uso del Modelo para Predicciones

### Predicción Simple

```python
from predictor_calibrado import PredictorCalibrado
import numpy as np

# Cargar modelo
predictor = PredictorCalibrado("modelos/random_forest_calibrado.pkl")

# Preparar features (ejemplo con las 30 features seleccionadas)
features = np.array([...])  # Tus 30 features

# Predecir
resultado = predictor.predecir(features)
print(f"Probabilidad: {resultado['probabilidad']*100:.1f}%")
print(f"Predicción: {'Gana' if resultado['prediccion'] == 1 else 'Pierde'}")
```

### Análisis de Apuesta

```python
# Analizar si vale la pena apostar
cuota = 2.50  # Cuota disponible
analisis = predictor.recomendar_apuesta(features, cuota, umbral_ev=0.08)

print(f"Decisión: {analisis['decision']}")
print(f"EV: {analisis['ev_porcentaje']:+.2f}%")
print(f"Ganancia esperada: {analisis['ganancia_esperada']:+.2f}€")
```

---

## 📊 Métricas del Modelo

### Calibración
- **Brier Score**: 0.1991 (< 0.20 ✅)
- **ECE**: 0.0222 (< 0.05 ✅)
- **Log Loss**: 0.5905
- **Accuracy**: 69.82%

### Backtesting (Umbral EV 8%)
- **ROI**: 57.41%
- **Win Rate**: 50.78%
- **Profit Factor**: 2.17
- **Max Drawdown**: -1.07%
- **Apuestas analizadas**: 1,030
- **Ganancia simulada**: +5,913€

---

## 🔧 Configuración Avanzada

### Cambiar Umbral de EV

Edita el umbral en `backtesting_fase2.py`:

```python
# Línea ~497
umbrales = [0.00, 0.03, 0.05, 0.08, 0.10]  # Añade más umbrales
```

### Usar Otro Modelo

Cambia el modelo en `backtesting_fase2.py`:

```python
# Línea ~689
modelo_path = "modelos/xgboost_calibrado.pkl"  # En lugar de random_forest
```

---

## 📚 Documentación Adicional

- **[FASE_2_RESULTADOS.md](FASE_2_RESULTADOS.md)** - Resultados detallados de Fase 2
- **[guiasProyecto/FASE_2_CALIBRACION.md](guiasProyecto/FASE_2_CALIBRACION.md)** - Guía de calibración
- **[guiasProyecto/FASE_3_OPTIMIZACION.md](guiasProyecto/FASE_3_OPTIMIZACION.md)** - Guía de optimización
- **[guiasProyecto/FASE_4_TRACKING.md](guiasProyecto/FASE_4_TRACKING.md)** - Guía de tracking

---

## 🐛 Troubleshooting

### Error: "No such file or directory: datos/..."

**Solución**: Ejecuta los pasos en orden. Primero descarga datos (Paso 1), luego procesa (Paso 2), etc.

### Error: "X has 149 features, but model expects 30"

**Solución**: Asegúrate de usar las features seleccionadas. Carga `resultados/selected_features.txt` para ver cuáles son.

### Error: "ModuleNotFoundError"

**Solución**: Instala dependencias:
```bash
pip install -r requirements.txt
```

### Modelos muy lentos

**Solución**: Reduce el tamaño del dataset o usa menos iteraciones en hyperparameter tuning.

---

## 🚀 Próximos Pasos (Fase 3)

- [ ] Optimización adicional de hiperparámetros
- [ ] Ensemble methods avanzados
- [ ] Feature engineering adicional
- [ ] Objetivo: Accuracy > 70%, Brier < 0.18

---

## 📝 Licencia

Este proyecto es de código abierto bajo licencia MIT.

---

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en GitHub.

---

**Última actualización**: Diciembre 2025  
**Versión**: 2.0 (Fase 2 Completada)
