"""
Script de Migración: Añadir Campos de Datos Enriquecidos
=========================================================

Este script migra la base de datos existente para añadir los nuevos campos
necesarios para el sistema de datos enriquecidos.

IMPORTANTE: Haz backup de tu base de datos antes de ejecutar este script.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backup_database(db_path: str) -> str:
    """Crea un backup de la base de datos"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    import shutil
    shutil.copy2(db_path, backup_path)
    logger.info(f"✅ Backup creado: {backup_path}")
    return backup_path


def migrate_database(db_path: str = "matches_v2.db"):
    """
    Migra la base de datos añadiendo los nuevos campos
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        logger.error(f"❌ Base de datos no encontrada: {db_path}")
        return False
    
    # Crear backup
    logger.info("📦 Creando backup de la base de datos...")
    backup_path = backup_database(str(db_path))
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("🔄 Iniciando migración...")
        
        # ============================================================
        # MIGRACIÓN 1: Añadir campos a tabla matches
        # ============================================================
        
        logger.info("📝 Añadiendo campos de tracking a tabla matches...")
        
        # Verificar si los campos ya existen
        cursor.execute("PRAGMA table_info(matches)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        new_columns = {
            "event_key": "VARCHAR(50)",
            "jugador1_key": "VARCHAR(50)",
            "jugador2_key": "VARCHAR(50)",
            "tournament_key": "VARCHAR(50)",
            "tournament_season": "VARCHAR(10)",
            "event_live": "VARCHAR(1) DEFAULT '0'",
            "event_qualification": "VARCHAR(10) DEFAULT 'False'",
            "event_final_result": "VARCHAR(20)",
        }
        
        for column_name, column_type in new_columns.items():
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE matches ADD COLUMN {column_name} {column_type}")
                    logger.info(f"  ✅ Añadido: matches.{column_name}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error añadiendo {column_name}: {e}")
            else:
                logger.info(f"  ℹ️  Ya existe: matches.{column_name}")
        
        # ============================================================
        # MIGRACIÓN 2: Crear índice para event_key
        # ============================================================
        
        logger.info("📝 Creando índice para event_key...")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_event_key ON matches(event_key)")
            logger.info("  ✅ Índice idx_matches_event_key creado")
        except Exception as e:
            logger.warning(f"  ⚠️  Error creando índice: {e}")
        
        # ============================================================
        # MIGRACIÓN 3: Añadir campo is_best a tabla odds_history
        # ============================================================
        
        logger.info("📝 Añadiendo campo is_best a tabla odds_history...")
        
        # Verificar si la tabla odds_history existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='odds_history'")
        if cursor.fetchone():
            cursor.execute("PRAGMA table_info(odds_history)")
            odds_columns = {row[1] for row in cursor.fetchall()}
            
            if "is_best" not in odds_columns:
                try:
                    cursor.execute("ALTER TABLE odds_history ADD COLUMN is_best BOOLEAN DEFAULT 0")
                    logger.info("  ✅ Añadido: odds_history.is_best")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error añadiendo is_best: {e}")
            else:
                logger.info("  ℹ️  Ya existe: odds_history.is_best")
            
            # Crear índice para bookmaker
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_bookmaker ON odds_history(bookmaker)")
                logger.info("  ✅ Índice idx_odds_bookmaker creado")
            except Exception as e:
                logger.warning(f"  ⚠️  Error creando índice: {e}")
        else:
            logger.info("  ℹ️  Tabla odds_history no existe (se creará automáticamente)")
        
        # ============================================================
        # COMMIT
        # ============================================================
        
        conn.commit()
        logger.info("✅ Migración completada exitosamente!")
        
        # Mostrar resumen
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN DE MIGRACIÓN")
        logger.info("="*60)
        logger.info(f"Base de datos: {db_path}")
        logger.info(f"Backup: {backup_path}")
        logger.info("\nCampos añadidos a 'matches':")
        for col in new_columns.keys():
            logger.info(f"  - {col}")
        logger.info("\nCampos añadidos a 'odds_history':")
        logger.info("  - is_best")
        logger.info("\nÍndices creados:")
        logger.info("  - idx_matches_event_key")
        logger.info("  - idx_odds_bookmaker")
        logger.info("="*60)
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error durante la migración: {e}")
        logger.error(f"💡 Puedes restaurar desde el backup: {backup_path}")
        return False


if __name__ == "__main__":
    print("🚀 Script de Migración - Datos Enriquecidos")
    print("=" * 60)
    print()
    print("Este script añadirá los siguientes campos:")
    print()
    print("📋 Tabla 'matches':")
    print("  - event_key (ID único en API-Tennis)")
    print("  - jugador1_key, jugador2_key (IDs de jugadores)")
    print("  - tournament_key (ID del torneo)")
    print("  - tournament_season (Temporada)")
    print("  - event_live (Estado en vivo)")
    print("  - event_qualification (Si es clasificación)")
    print("  - event_final_result (Resultado final)")
    print()
    print("📋 Tabla 'odds_history':")
    print("  - is_best (Marca la mejor cuota)")
    print()
    print("⚠️  IMPORTANTE: Se creará un backup automático antes de migrar")
    print()
    
    respuesta = input("¿Continuar con la migración? (s/n): ")
    
    if respuesta.lower() in ['s', 'si', 'y', 'yes']:
        print()
        success = migrate_database()
        
        if success:
            print()
            print("✅ ¡Migración completada exitosamente!")
            print()
            print("🚀 Próximos pasos:")
            print("  1. Reiniciar la API para cargar los cambios")
            print("  2. Los nuevos partidos se crearán con los campos adicionales")
            print("  3. El endpoint GET /matches devolverá datos enriquecidos")
        else:
            print()
            print("❌ La migración falló. Revisa los logs arriba.")
            print("💡 Puedes restaurar desde el backup si es necesario.")
    else:
        print()
        print("❌ Migración cancelada.")
