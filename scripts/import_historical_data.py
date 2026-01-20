
"""
Script de importación de datos históricos a la base de datos SQLite.
Lee los archivos CSV de Jeff Sackmann (tennis_atp) y los inserta en la tabla matches.
"""
import pandas as pd
import sqlite3
import glob
import os
from datetime import datetime
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "matches_v2.db"
DATA_PATH = "datos/raw"

def connect_db():
    return sqlite3.connect(DB_PATH)

def normalize_date(date_int):
    """Convierte fecha YYYYMMDD (int) a YYYY-MM-DD (str)"""
    try:
        date_str = str(int(date_int))
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    except:
        return None

def import_csv_to_db():
    conn = connect_db()
    cursor = conn.cursor()
    
    # Asegurar que la tabla existe (por si acaso se ejecuta antes de la app)
    # Nota: Idealmente el schema ya debería estar creado por la app
    
    csv_files = glob.glob(os.path.join(DATA_PATH, "atp_matches_*.csv"))
    csv_files.sort()
    
    if not csv_files:
        logger.error(f"❌ No se encontraron archivos CSV en {DATA_PATH}")
        return

    total_inserted = 0
    
    for file in csv_files:
        logger.info(f"📂 Procesando {file}...")
        try:
            df = pd.read_csv(file)
            
            # Columnas esperadas en CSV de Sackmann:
            # tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, 
            # match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand, 
            # winner_ht, winner_ioc, winner_age, loser_id, loser_seed, loser_entry, 
            # loser_name, loser_hand, loser_ht, loser_ioc, loser_age, score, best_of, 
            # round, minutes, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, 
            # w_SvGms, w_bpSaved, w_bpFaced, l_ace, l_df, l_svpt, l_1stIn, l_1stWon, 
            # l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced, winner_rank, winner_rank_points, 
            # loser_rank, loser_rank_points
            
            entries = []
            file_inserted = 0
            
            for _, row in df.iterrows():
                try:
                    fecha = normalize_date(row['tourney_date'])
                    if not fecha:
                        continue
                        
                    # Mapeo de columnas
                    match_data = {
                        'fecha_partido': fecha,  # Aproximada (fecha torneo)
                        'hora_inicio': '00:00',
                        'torneo': row['tourney_name'],
                        'tournament_season': str(row['tourney_date'])[:4],
                        'ronda': row['round'],
                        'superficie': row['surface'] if isinstance(row['surface'], str) else 'Hard', # Default
                        'jugador1_nombre': row['winner_name'], # Ganador siempre como J1 temporalmente? No, mejor aleatorio o fijo. 
                        # En este modelo simple, pongamos Winner como J1 y Loser como J2, 
                        # PERO marcaremos el resultado.
                        'jugador1_ranking': row['winner_rank'] if pd.notna(row['winner_rank']) else 0,
                        'jugador2_nombre': row['loser_name'],
                        'jugador2_ranking': row['loser_rank'] if pd.notna(row['loser_rank']) else 0,
                        'estado': 'completado',
                        'resultado_ganador': row['winner_name'],
                        'resultado_marcador': row['score'],
                        'event_final_result': '2-0' if 'retired' not in str(row['score']) else 'ret', # Simplificado
                    }
                    
                    # Insertar evitando duplicados
                    # Usamos INSERT OR IGNORE basándonos en la constraint UNIQUE(fecha, j1, j2)
                    # Nota: La fecha csv es fecha de torneo ("lunes"), no fecha exacta del partido.
                    # Esto podría causar conflictos si jugaron 2 veces en el mismo torneo (raro, round robin)
                    # Pero para histórico sirve.
                    
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
                    
                    if cursor.rowcount > 0:
                        file_inserted += 1
                        
                except Exception as e:
                    # logger.warning(f"Error fila: {e}")
                    continue
            
            conn.commit()
            logger.info(f"✅ Insertados {file_inserted} partidos de {file}")
            total_inserted += file_inserted
            
        except Exception as e:
            logger.error(f"❌ Error leyendo {file}: {e}")
            
    conn.close()
    logger.info(f"🎉 Importación completada. Total partidos importados: {total_inserted}")

if __name__ == "__main__":
    import_csv_to_db()
