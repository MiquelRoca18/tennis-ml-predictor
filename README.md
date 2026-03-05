# Tenly — API Backend

API REST del proyecto **Tenly**. Expone un backend en **FastAPI** que:

- Gestiona partidos de tenis (pendientes, en juego, completados) sobre una base de datos SQLite/PostgreSQL.
- Calcula predicciones con un **baseline ELO + mercado** (sin modelo ML `.pkl`).
- Devuelve probabilidad, valor esperado (EV), decisión (apostar/no apostar) y stake recomendado (Kelly fraccional).
- Se despliega en **Railway** usando `Dockerfile`, `start.sh` y `railway.json`.

## 📚 Documentación

- **Manual de usuario (API):** `MANUAL_USUARIO.md`
- **Documentación técnica (arquitectura, modelo, Railway, frontend):** `DOCUMENTACION_TECNICA.md`

## 🚀 Puesta en marcha en local

### Requisitos

- Python 3.8+
- `pip`
- (Opcional) virtualenv (`python -m venv venv`)

### Pasos

```bash
# 1. Clonar repositorio
git clone &lt;URL_DEL_REPO_BACKEND&gt;
cd tennis-ml-predictor

# 2. (Opcional) Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.template .env
# Editar .env y añadir al menos API_TENNIS_API_KEY (api-tennis.com)

# 5. Levantar la API
uvicorn src.api.api_v2:app --reload --host 0.0.0.0 --port 8000
```

- Documentación interactiva disponible en `http://localhost:8000/docs` y `http://localhost:8000/redoc`.
- Si no se define `DATABASE_URL`, se usa **SQLite** (`matches_v2.db`). En Railway se usa PostgreSQL vía `DATABASE_URL`.

## 🧠 Modelo de predicción (resumen)

- **Sin modelo ML**: las predicciones usan solo:
  - Probabilidad ELO (`prob_elo`) calculada desde el histórico de partidos.
  - Probabilidad implícita de la cuota (`prob_mercado = 1 / cuota`).
- **Combinación:** `prob_j1 = 0.6 × prob_elo + 0.4 × prob_mercado` (configurable con `BASELINE_ELO_PESO`).
- **EV:** `EV = prob_j1 × cuota − 1`.
- **Decisión:** APOSTAR si `EV &gt; EV_THRESHOLD`.
- **Stake:** criterio de Kelly fraccional con límites (mínimo, máximo % del bankroll, máximo €/apuesta).

Para el detalle completo (glosario, fórmulas y ejemplo numérico) ver la sección 2.2 de `DOCUMENTACION_TECNICA.md`.

## 🧱 Estructura principal

```text
tennis-ml-predictor/
├── src/
│   ├── api/            # FastAPI (api_v2, rutas de detalle)
│   ├── config/         # Configuración centralizada (Config)
│   ├── database/       # MatchDatabase (SQLite/PostgreSQL)
│   ├── prediction/     # PredictorCalibrado, FeatureGeneratorService
│   ├── features/       # ELO, forma, H2H, superficie, etc.
│   ├── services/       # API-Tennis, odds, stats, torneos, players…
│   ├── automation/     # Jobs de detección/sync de partidos
│   └── utils/          # Utilidades (Kelly, logs, etc.)
│
├── datos/              # CSV de histórico (TML), cache, etc.
├── scripts/            # Backtesting y utilidades de mantenimiento
├── tests/              # Tests de API y sistema
├── DOCUMENTACION_TECNICA.md
├── MANUAL_USUARIO.md
├── Dockerfile
├── railway.json
└── start.sh
```

## 🛠 Despliegue en Railway (resumen)

- **Build:** Railway usa `Dockerfile` para construir la imagen.
- **Start:** `railway.json` ejecuta `./start.sh` (`uvicorn src.api.api_v2:app ...`).
- **Variables críticas:**
  - `DATABASE_URL` (PostgreSQL de Railway).
  - `API_TENNIS_API_KEY` (api-tennis.com).

Consulta `DOCUMENTACION_TECNICA.md` para el flujo detallado de jobs y predicciones en producción.

