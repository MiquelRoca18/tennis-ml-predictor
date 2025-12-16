# 🎾 Tennis ML Predictor

Sistema de predicción de partidos de tenis usando Machine Learning con probabilidades calibradas, optimizado para apuestas deportivas inteligentes.

## 📊 Resultados del Sistema

- **Accuracy**: 71.57% (modelo calibrado)
- **Brier Score**: 0.1914 (calibración excelente)
- **ECE**: 0.0474 (casi perfecta calibración)
- **ROI Backtesting**: 57.41%
- **Kelly Criterion**: +96% ROI vs Flat Betting
- **Line Shopping**: +0.5-2% EV adicional
- **Datos**: 2022-2025 (TML Database)
- **Features**: 30 seleccionadas de 149 generadas

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8+
- pip
- Git

### Instalación y Configuración

```bash
# 1. Clonar repositorio
git clone https://github.com/MiquelRoca18/tennis-ml-predictor.git
cd tennis-ml-predictor

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno (opcional, para bookmakers y email)
cp .env.template .env
# Editar .env con tus credenciales
```

### Ejecutar Pipeline Completo

```bash
# Pipeline completo: descarga datos + entrena + valida
python setup_and_train.py --full
```

**⏱️ Tiempo**: 30-40 minutos  
**✅ Resultado**: Modelo entrenado y validado, listo para usar

### Opciones Alternativas

```bash
# Solo entrenar (si ya tienes datos)
python setup_and_train.py --train-only

# Solo validar (si ya tienes modelo)
python setup_and_train.py --validate-only
```

---

## 💡 ¿Qué Hace Este Sistema?

Este proyecto es un **sistema completo de predicción de tenis** que:

1. **Descarga datos** históricos de partidos (2022-2025)
2. **Genera 149 features** avanzadas (ELO, forma reciente, H2H, superficie, etc.)
3. **Entrena 4 modelos** ML (Random Forest, Gradient Boosting, Logistic Regression, XGBoost)
4. **Calibra probabilidades** para apuestas (Isotonic + Platt Scaling)
5. **Optimiza apuestas** con Kelly Criterion
6. **Compara cuotas** de múltiples bookmakers (line shopping)
7. **Automatiza predicciones** diarias con alertas por email

**Resultado**: Predicciones calibradas con ventaja estadística para apuestas inteligentes.

---

## 📁 Estructura del Proyecto

```
tennis-ml-predictor/
├── validate.py              # Validación unificada (todas las fases)
├── demo.py                  # Demos del sistema
├── setup_and_train.py       # Pipeline maestro
├── predictor_calibrado.py   # Predictor principal
│
├── scripts/
│   └── internal/            # Scripts de uso ocasional
│
├── src/
│   ├── config/              # Configuración centralizada
│   ├── utils/               # Utilidades compartidas
│   ├── data/                # Descarga y procesamiento de datos
│   ├── features/            # Feature engineering (ELO, H2H, etc.)
│   ├── models/              # Entrenamiento y optimización
│   ├── prediction/          # Sistema de predicción
│   ├── betting/             # Kelly Criterion + Bankroll
│   ├── bookmakers/          # Line shopping + Alertas
│   ├── tracking/            # Tracking de apuestas
│   ├── automation/          # Automatización 24/7
│   ├── validation/          # Validaciones refactorizadas
│   └── demos/               # Demos refactorizadas
│
├── datos/                   # Datasets y base de datos
├── modelos/                 # Modelos entrenados
└── resultados/              # Reportes y análisis
```

---

## 🎯 Uso del Sistema

### 1. Validar el Sistema

```bash
# Validar todas las fases
python validate.py --all

# Validar fase específica
python validate.py --phase 2  # Calibración
python validate.py --phase 5  # Kelly Criterion
python validate.py --phase 7  # Automatización
```

### 2. Ejecutar Demos

```bash
# Todas las demos
python demo.py --all

# Demo específica
python demo.py --feature tracking
python demo.py --feature kelly
python demo.py --feature bookmakers
```

### 3. Hacer Predicciones

```python
from predictor_calibrado import PredictorCalibrado

# Cargar modelo
predictor = PredictorCalibrado('modelos/production/random_forest_calibrado.pkl')

# Predecir partido
prob = predictor.predecir_partido(
    jugador1="Djokovic",
    jugador2="Nadal",
    superficie="Clay"
)

print(f"Probabilidad Djokovic: {prob:.2%}")
```

### 4. Sistema de Tracking

```python
from src.tracking import TrackingSystem

# Inicializar tracking con Kelly
sistema = TrackingSystem(
    modelo_path='modelos/production/random_forest_calibrado.pkl',
    bankroll_actual=1000,
    usar_kelly=True
)

# Registrar predicción
sistema.predecir_y_registrar(
    jugador1="Federer",
    jugador2="Murray",
    cuota=2.10
)

# Generar reporte
sistema.generar_reporte()
```

---

## 🔧 Configuración Avanzada

### Variables de Entorno (.env)

```bash
# API de Bookmakers (opcional)
ODDS_API_KEY=tu_api_key_aqui

# Email para alertas (opcional)
EMAIL_USER=tu_email@gmail.com
EMAIL_PASSWORD=tu_app_password

# Parámetros del sistema
MIN_BET=5
MAX_BET_PCT=5
KELLY_FRACTION=0.25
```

### Personalización

- **Modelos**: Editar `src/models/hyperparameter_tuning.py`
- **Features**: Añadir en `src/features/`
- **Bookmakers**: Configurar en `src/config/settings.py`

---

## 📊 Fases del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1 | Datos y Preprocesamiento | ✅ |
| 2 | Calibración de Modelos | ✅ |
| 3 | Feature Engineering + Optimización | ✅ |
| 4 | Sistema de Tracking | ✅ |
| 5 | Kelly Criterion | ✅ |
| 6 | Múltiples Bookmakers | ✅ |
| 7 | Automatización 24/7 | ✅ |

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

---

## APIs

- **TML Database** por los datos de partidos
- **The Odds API** por las cuotas de bookmakers
- Comunidad de ML y apuestas deportivas


