# 🎾 Tenly API (Backend)

API REST del proyecto **Tenly**: predicción de partidos de tenis para apuestas deportivas. Usa **baseline ELO + mercado** (60% ELO + 40% probabilidad implícita de la cuota), con EV, Kelly y filtros conservadores.

**Documentación:** [Manual de usuario](docs/MANUAL_USUARIO.md) · [Documentación técnica](docs/DOCUMENTACION_TECNICA.md)

## 📊 Estrategia en producción

- **Baseline**: 60% ELO + 40% mercado (sin modelo ML)
- **Filtros**: EV > 10%, cuota < 2.0, probabilidad > 70%
- **Backtesting**: ROI estable 4/4 años con esta configuración

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

### Ejecutar la API

```bash
# Levantar la API (predicciones con baseline ELO + mercado)
uvicorn src.api.api_v2:app --reload --host 0.0.0.0 --port 8000
```

---

## 💡 ¿Qué hace este sistema?

1. **Predicciones** con baseline: probabilidad = 60% ELO + 40% mercado (cuota)
2. **ELO y features** desde histórico en BD (FeatureGeneratorService)
3. **EV, Kelly y filtros** (min prob, max cuota) para recomendaciones de apuesta
4. **API** para crear partidos, obtener predicciones y listar partidos del día
5. **Automatización** opcional: fetch diario de partidos, actualización de datos

---

## 📁 Estructura del Proyecto

```
tennis-ml-predictor/
├── scripts/
│   ├── backtesting_produccion_real_completo.py  # Backtesting (baseline ELO)
│   └── internal/            # Scripts de uso ocasional
│
├── src/
│   ├── config/              # Configuración centralizada
│   ├── api/                  # API FastAPI
│   ├── prediction/          # Predictor baseline + FeatureGeneratorService
│   ├── features/            # ELO, forma, H2H, superficie
│   ├── utils/               # Utilidades compartidas
│   ├── automation/          # DataUpdater, daily match fetcher
│   └── services/             # Odds, predicciones, etc.
│
├── datos/                   # Datos y cache
└── resultados/              # Opcional (backtesting)
```

---

## 🎯 Uso

### Predicciones (baseline ELO + mercado)

```python
from src.prediction.predictor_calibrado import PredictorCalibrado
from src.config.settings import Config

predictor = PredictorCalibrado(Config.MODEL_PATH)
resultado = predictor.predecir_partido(
    jugador1="Djokovic",
    jugador2="Nadal",
    superficie="Clay",
    cuota=2.10
)
# resultado["probabilidad"], resultado["expected_value"], resultado["decision"], etc.

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
# API-Tennis (obligatoria para partidos y cuotas)
API_TENNIS_API_KEY=tu_api_key_api_tennis

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
- **API-Tennis y parámetros**: Configurar en `src/config/settings.py` y `.env`

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

- **API-Tennis** (api-tennis.com): partidos, cuotas, resultados y rankings.
- **TML** (stats.tennismylife.org): CSVs de histórico para el cálculo de ELO.


