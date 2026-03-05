# Manual de usuario — Tenly API (Backend)

Manual básico para usar la API de **Tenly**: comprobar estado, obtener partidos y predicciones.

---

## 1. ¿Qué es el backend?

Es la **API REST** que sirve partidos, predicciones y datos de tenis. La app **Tenly** (frontend) se conecta a esta API. También puedes usarla desde el navegador, Postman o cualquier cliente HTTP.

**URL en producción:** `https://tennis-ml-predictor-production.up.railway.app`

**Documentación interactiva:**  
- Swagger: `https://tennis-ml-predictor-production.up.railway.app/docs`  

---

## 2. Comprobar que la API funciona

- **GET /** — Información básica de la API (nombre, versión, enlaces).
- **GET /health** — Estado del servicio (OK si la API y la base de datos responden).
- **GET /keepalive** — Respuesta rápida "alive"; útil para que el servidor no se duerma (p. ej. cron externo).

Si `/health` devuelve `"status": "ok"`, la API está operativa.

---

## 3. Partidos

- **GET /matches?date=YYYY-MM-DD** — Lista de partidos para una fecha (ej: `?date=2025-03-04`).
- **GET /matches/upcoming** — Partidos próximos.
- **GET /matches/{id}/details** — Detalle completo de un partido (marcador, predicción, cuotas, estadísticas, etc.).

Tenly usa estos endpoints para mostrar el listado y el detalle de cada partido.

---

## 4. Predicciones

- **POST /matches/predict** — Genera una predicción para un partido. Se envía en el cuerpo (JSON): jugadores, superficie, cuota, fecha, etc. La API devuelve probabilidad, EV, decisión (APOSTAR/NO APOSTAR) y stake recomendado.

Las predicciones también se generan **automáticamente** en el servidor cuando se detectan partidos nuevos con cuotas (jobs cada 4 h y al arrancar). No hace falta llamar a `/matches/predict` manualmente para los partidos que ya están en Tenly.

---

## 5. Resultados y actualización

- **PUT /matches/{id}/result** — Registrar el resultado de un partido (ganador, sets).
- **POST /matches/{id}/refresh** — Pedir al servidor que actualice ese partido (resultado, cuotas) desde la API de tenis.

Tenly usa "Actualizar resultados" en Mis apuestas para llamar a endpoints de batch que a su vez actualizan resultados en el backend.

---

## 6. Configuración de apuestas (opcional)

- **GET /settings/betting** — Obtener configuración de apuestas (p. ej. bankroll).
- **PATCH /settings/betting** — Actualizar bankroll u otros parámetros (según implementación).

Si usas Tenly sin cuenta, el bankroll puede guardarse aquí; con cuenta, puede sincronizarse con Supabase.

---

## 7. Más endpoints (referencia)

A continuación se listan el resto de endpoints que expone la API, útiles para torneos, perfiles de jugador, clasificaciones, búsqueda y datos extra de partidos. La documentación interactiva en `/docs` muestra parámetros y respuestas.

**Partidos (adicionales)**  
- **GET /matches/range** — Partidos en un rango de fechas.  
- **GET /matches/{id}/analysis** — Análisis del partido (predicción, EV, etc.).  
- **GET /matches/{id}/history** — Historial de encuentros entre los dos jugadores (H2H).

**Jugadores**  
- **GET /players/lookup** — Búsqueda de jugadores por texto (query).  
- **GET /players/{player_key}** — Perfil de un jugador (datos básicos).  
- **GET /players/{player_key}/matches** — Partidos del jugador.  
- **GET /players/{player_key}/upcoming** — Próximos partidos del jugador.  
- **GET /players/{player_key}/stats** — Estadísticas del jugador (por temporada, superficie, etc.).

**Head-to-head (H2H)**  
- **GET /h2h/{player1_key}/{player2_key}** — Enfrentamientos directos entre dos jugadores.  
- **GET /h2h/match/{match_id}** — H2H en el contexto de un partido concreto.

**Clasificaciones**  
- **GET /rankings/{league}** — Ranking de una liga/circuito (ej.: atp, wta).  
- **GET /rankings/player/{player_key}** — Posición en el ranking de un jugador.

**Torneos**  
- **GET /tournaments** — Lista de torneos.  
- **GET /tournaments/{tournament_key}** — Detalle de un torneo.  
- **GET /tournaments/{tournament_key}/matches** — Partidos de un torneo.

**Detalle de partido (timeline, stats, cuotas, H2H)**  
- **GET /matches/{id}/details** — Detalle completo (recomendado).  
- **GET /matches/{id}/full** — Mismo contenido que `details`.  
- **GET /matches/{id}/timeline** — Timeline de juegos.  
- **GET /matches/{id}/stats** — Estadísticas del partido.  
- **GET /matches/{id}/pbp** — Punto por punto.  
- **GET /matches/{id}/odds** — Cuotas de bookmakers.  
- **GET /matches/{id}/h2h** — Head-to-head entre los jugadores.

Para crons externos (p. ej. cron-job.org) se usan **GET /admin/trigger-retraining** (sync cuotas y predicciones), **GET /cron/refresh-elo** (actualizar datos ELO) y **GET /admin/cron-status** (ver último resultado de cada cron). El resto de la API se documenta en `/docs`.

---

## 8. Uso típico (resumen)

| Objetivo              | Acción |
|-----------------------|--------|
| Saber si la API está bien | `GET /health` |
| Ver partidos de un día    | `GET /matches?date=YYYY-MM-DD` |
| Ver detalle de un partido | `GET /matches/{id}/details` |
| Generar predicción a mano | `POST /matches/predict` con JSON |
| Ver documentación        | Abrir `/docs` o `/redoc` en el navegador |

---

## 9. Errores frecuentes

- **Timeout o sin respuesta:** El servidor (Railway) puede estar "dormido"; llama a `/keepalive` o espera a que la siguiente petición lo reactive.
- **404 en un partido:** El ID no existe o la fecha no tiene partidos; comprueba la fecha y el ID.
- **500 en predicción:** Falta configuración (p. ej. API de tenis) o datos (ELO); revisa variables de entorno y logs en Railway.

Para desarrollo local, la API suele estar en `http://localhost:8000` (mismos paths: `/health`, `/matches`, etc.).

---

## 10. Ejecutar la API en local

Si quieres descargar el código del backend y ejecutar la API en tu ordenador (por ejemplo para desarrollo o para que un profesor pueda reproducir el proyecto):

1. **Clonar el repositorio** del backend (carpeta `tennis-ml-predictor` del proyecto Tenly).
2. **Requisitos:** Python 3.8+ y pip.
3. **Entorno virtual (recomendado):** `python -m venv venv` y activarlo (`source venv/bin/activate` en Linux/Mac; `venv\Scripts\activate` en Windows).
4. **Instalar dependencias:** `pip install -r requirements.txt`.
5. **Variables de entorno:** Copiar `.env.template` a `.env`.
   - **Sin `DATABASE_URL`:** la API usará **SQLite** en local (no hace falta base de datos en la nube).
   - **Para que haya partidos y datos reales** es necesario configurar `API_TENNIS_API_KEY` (clave de api-tennis.com). Sin esta variable la API arranca y responde, pero la lista de partidos estará vacía porque los partidos se obtienen de esa API externa. Tras arrancar, los jobs en segundo plano irán rellenando la base de datos; la primera vez puede tardar unos minutos.
   - Opcional: si quieres mejor calidad en las predicciones (ELO), puedes colocar CSVs de histórico en la carpeta `datos/raw/` (por ejemplo los que se descargan en el Dockerfile desde stats.tennismylife.org). Si no hay CSVs, el ELO se calcula solo con los partidos completados que vaya teniendo la BD.
6. **Arrancar la API:** `uvicorn src.api.api_v2:app --reload --host 0.0.0.0 --port 8000`. La API quedará en `http://localhost:8000`; la documentación en `http://localhost:8000/docs`.

En local no se usa Railway: el servidor y la base de datos (SQLite) corren en tu máquina. Para más detalle (variables opcionales, Docker, despliegue), consulta la documentación técnica del proyecto o el README del repositorio.
