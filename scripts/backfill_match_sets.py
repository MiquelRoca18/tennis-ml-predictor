#!/usr/bin/env python3
"""
Script para crear sets estructurados desde marcadores existentes
Parsea resultado_marcador y crea registros en match_sets
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.match_database import MatchDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_score_to_sets(marcador: str):
    """
    Parsea marcador como "6-4, 7-5, 6-3" a lista de sets
    
    Returns:
        List[Dict]: [{"set": 1, "p1": 6, "p2": 4}, ...]
    """
    sets = []
    if not marcador or marcador == '-' or marcador == '':
        return sets
    
    # Limpiar espacios y separar por comas
    parts = [p.strip() for p in marcador.split(',')]
    
    set_number = 1
    for part in parts:
        # Buscar TODOS los patrones número-número en cada parte
        matches = re.findall(r'(\d+)-(\d+)', part)
        
        for match in matches:
            try:
                p1_score = int(match[0])
                p2_score = int(match[1])
                
                # Validar que sean scores válidos de tenis (0-7, permitir tiebreak)
                if 0 <= p1_score <= 7 and 0 <= p2_score <= 7:
                    sets.append({
                        "set": set_number,
                        "p1": p1_score,
                        "p2": p2_score
                    })
                    set_number += 1
            except (ValueError, IndexError):
                logger.warning(f"Error parseando score: {match}")
    
    return sets

def backfill_match_sets():
    """Crea sets estructurados desde marcadores existentes"""
    
    logger.info("=" * 80)
    logger.info("🎾 BACKFILL DE SETS ESTRUCTURADOS")
    logger.info("=" * 80)
    
    db = MatchDatabase()
    cursor = db.conn.cursor()
    
    # Verificar que la tabla existe
    table_exists = cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='match_sets'
    """).fetchone()
    
    if not table_exists:
        logger.error("❌ Tabla match_sets no existe. Ejecutar schema v7 primero.")
        db.close()
        return
    
    # Paso 1: Obtener partidos con marcador
    logger.info("\n📝 Paso 1: Obteniendo partidos con marcador...")
    
    matches = cursor.execute("""
        SELECT id, resultado_marcador, jugador1_nombre, jugador2_nombre
        FROM matches
        WHERE resultado_marcador IS NOT NULL
        AND resultado_marcador != ''
        AND resultado_marcador != '-'
    """).fetchall()
    
    logger.info(f"Encontrados {len(matches)} partidos con marcador")
    
    # Paso 2: Parsear y guardar sets
    logger.info("\n🔄 Paso 2: Parseando y guardando sets...")
    
    total_sets = 0
    matches_processed = 0
    matches_skipped = 0
    
    for match in matches:
        match_id = match['id']
        marcador = match['resultado_marcador']
        
        # Parsear marcador
        sets = parse_score_to_sets(marcador)
        
        if not sets:
            matches_skipped += 1
            logger.debug(f"Skipped match {match_id}: no se pudo parsear '{marcador}'")
            continue
        
        # Guardar sets
        for set_data in sets:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO match_sets (
                        match_id, set_number, player1_score, player2_score
                    ) VALUES (?, ?, ?, ?)
                """, (match_id, set_data['set'], set_data['p1'], set_data['p2']))
                total_sets += 1
            except Exception as e:
                logger.error(f"Error guardando set para match {match_id}: {e}")
        
        matches_processed += 1
        
        if matches_processed % 100 == 0:
            logger.info(f"  Procesados {matches_processed}/{len(matches)} partidos...")
    
    db.conn.commit()
    
    # Paso 3: Verificar resultados
    logger.info("\n📈 Paso 3: Verificando resultados...")
    
    total_matches = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    
    matches_with_sets = cursor.execute("""
        SELECT COUNT(DISTINCT match_id) FROM match_sets
    """).fetchone()[0]
    
    total_sets_db = cursor.execute("SELECT COUNT(*) FROM match_sets").fetchone()[0]
    
    logger.info(f"\nResultados:")
    logger.info(f"  Total de partidos: {total_matches}")
    logger.info(f"  Partidos procesados: {matches_processed}")
    logger.info(f"  Partidos con sets: {matches_with_sets}/{total_matches} ({matches_with_sets/total_matches*100:.1f}%)")
    logger.info(f"  Total de sets creados: {total_sets_db}")
    logger.info(f"  Promedio sets/partido: {total_sets_db/matches_with_sets:.1f}" if matches_with_sets > 0 else "  N/A")
    logger.info(f"  Partidos skipped: {matches_skipped}")
    
    logger.info("\n✅ Backfill completado!")
    
    db.close()

if __name__ == "__main__":
    backfill_match_sets()
