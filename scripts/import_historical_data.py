
"""
Script de importación de datos históricos a la base de datos.

Lee CSVs en formato TML-Database / Sackmann (winner_name, loser_name, tourney_date, etc.)
y los inserta en la tabla matches como partidos completados.

- Con DATABASE_URL (PostgreSQL, p. ej. Railway): importa a esa BD.
- Sin DATABASE_URL: usa SQLite local (matches_v2.db).

Uso para Railway:
  1. Descargar CSVs de https://github.com/Tennismylife/TML-Database (p. ej. 2022.csv, 2023.csv, 2024.csv)
  2. Guardarlos como datos/raw/atp_matches_2022_tml.csv, atp_matches_2023_tml.csv, etc.
  3. DATABASE_URL="postgresql://..." python scripts/import_historical_data.py
"""
import pandas as pd
import sqlite3
import glob
import os
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "matches_v2.db")
DATA_PATH = os.environ.get("DATA_PATH", "datos/raw")
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():
    """Conexión a SQLite o PostgreSQL según DATABASE_URL."""
    if DATABASE_URL:
        try:
            import psycopg2
            # Railway a veces inyecta postgres://; psycopg2 quiere postgresql://
            url = DATABASE_URL if DATABASE_URL.startswith("postgresql") else DATABASE_URL.replace("postgres://", "postgresql://", 1)
            return psycopg2.connect(url), "pg"
        except Exception as e:
            logger.error(f"❌ Error conectando a PostgreSQL: {e}")
            raise
    return sqlite3.connect(DB_PATH), "sqlite"

def normalize_date(date_int):
    """Convierte fecha YYYYMMDD (int) a YYYY-MM-DD (str)"""
    try:
        date_str = str(int(date_int))
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    except:
        return None

def _insert_match_sqlite(cursor, match_data):
    cursor.execute("""
        INSERT OR IGNORE INTO matches (
            fecha_partido, hora_inicio, torneo, tournament_season, ronda, superficie,
            jugador1_nombre, jugador1_ranking, jugador2_nombre, jugador2_ranking,
            estado, resultado_ganador, resultado_marcador, event_final_result
        ) VALUES (
            :fecha_partido, :hora_inicio, :torneo, :tournament_season, :ronda, :superficie,
            :jugador1_nombre, :jugador1_ranking, :jugador2_nombre, :jugador2_ranking,
            :estado, :resultado_ganador, :resultado_marcador, :event_final_result
        )
    """, match_data)
    return cursor.rowcount


def _insert_match_postgres(cursor, match_data):
    cursor.execute("""
        INSERT INTO matches (
            fecha_partido, hora_inicio, torneo, tournament_season, ronda, superficie,
            jugador1_nombre, jugador1_ranking, jugador2_nombre, jugador2_ranking,
            estado, resultado_ganador, resultado_marcador, event_final_result
        ) VALUES (
            %(fecha_partido)s, %(hora_inicio)s, %(torneo)s, %(tournament_season)s, %(ronda)s, %(superficie)s,
            %(jugador1_nombre)s, %(jugador1_ranking)s, %(jugador2_nombre)s, %(jugador2_ranking)s,
            %(estado)s, %(resultado_ganador)s, %(resultado_marcador)s, %(event_final_result)s
        )
        ON CONFLICT (fecha_partido, jugador1_nombre, jugador2_nombre) DO NOTHING
    """, match_data)
    return cursor.rowcount


def import_csv_to_db():
    conn, db_type = get_connection()
    cursor = conn.cursor()
    if db_type == "pg":
        logger.info("📂 Conectado a PostgreSQL (DATABASE_URL)")
    else:
        logger.info("📂 Conectado a SQLite")

    # Aceptar atp_matches_*.csv (ej. atp_matches_2024_tml.csv) o TML directo: 2022.csv, 2023.csv, 2024.csv
    csv_files = glob.glob(os.path.join(DATA_PATH, "atp_matches_*.csv"))
    for year in (2022, 2023, 2024, 2025, 2026):
        path = os.path.join(DATA_PATH, f"{year}.csv")
        if path not in csv_files and os.path.isfile(path):
            csv_files.append(path)
    csv_files.sort()

    if not csv_files:
        logger.error(
            f"❌ No se encontraron CSV en {DATA_PATH}. "
            "Pon atp_matches_*_tml.csv o descarga de TML-Database como 2022.csv, 2024.csv, etc."
        )
        return

    insert_fn = _insert_match_postgres if db_type == "pg" else _insert_match_sqlite
    total_inserted = 0

    for file in csv_files:
        logger.info(f"📂 Procesando {file}...")
        try:
            df = pd.read_csv(file)

            # Columnas TML-Database / Sackmann: tourney_date, winner_name, loser_name, score, surface, round, etc.

            file_inserted = 0
            for _, row in df.iterrows():
                try:
                    fecha = normalize_date(row["tourney_date"])
                    if not fecha:
                        continue

                    match_data = {
                        "fecha_partido": fecha,
                        "hora_inicio": "00:00",
                        "torneo": row["tourney_name"],
                        "tournament_season": str(row["tourney_date"])[:4],
                        "ronda": row["round"],
                        "superficie": row["surface"] if isinstance(row["surface"], str) else "Hard",
                        "jugador1_nombre": row["winner_name"],
                        "jugador1_ranking": int(row["winner_rank"]) if pd.notna(row.get("winner_rank")) else 0,
                        "jugador2_nombre": row["loser_name"],
                        "jugador2_ranking": int(row["loser_rank"]) if pd.notna(row.get("loser_rank")) else 0,
                        "estado": "completado",
                        "resultado_ganador": row["winner_name"],
                        "resultado_marcador": str(row["score"]) if pd.notna(row.get("score")) else "",
                        "event_final_result": "2-0" if "retired" not in str(row.get("score", "")).lower() else "ret",
                    }

                    if insert_fn(cursor, match_data) > 0:
                        file_inserted += 1
                except Exception:
                    continue

            conn.commit()
            logger.info(f"✅ Insertados {file_inserted} partidos de {file}")
            total_inserted += file_inserted

        except Exception as e:
            logger.error(f"❌ Error leyendo {file}: {e}")

    cursor.close()
    conn.close()
    logger.info(f"🎉 Importación completada. Total partidos importados: {total_inserted}")

if __name__ == "__main__":
    import_csv_to_db()
