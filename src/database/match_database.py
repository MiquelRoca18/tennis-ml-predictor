"""
Match Database Manager - Tennis ML Predictor v2.0
==================================================

Gestión de base de datos para partidos con predicciones versionadas,
tracking de apuestas y estadísticas.

Supports both SQLite (local development) and PostgreSQL (Railway production).
"""

import sqlite3
import os
from pathlib import Path
from datetime import date, timedelta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MatchDatabase:
    """
    Gestor de base de datos para partidos de tenis con predicciones y apuestas
    
    Automatically detects DATABASE_URL environment variable and uses PostgreSQL if available,
    otherwise falls back to SQLite for local development.
    """

    def __init__(self, db_path: str = "matches_v2.db"):
        """
        Inicializa la conexión a la base de datos
        
        Checks for DATABASE_URL environment variable (Railway PostgreSQL).
        If found, uses PostgreSQL. Otherwise uses SQLite.

        Args:
            db_path: Ruta al archivo de base de datos SQLite (usado solo si no hay DATABASE_URL)
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.is_postgres = False
        self.engine = None
        
        # Check for PostgreSQL URL (Railway)
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            # PostgreSQL mode (Railway production)
            self._init_postgres(database_url)
        else:
            # SQLite mode (local development)
            self._init_sqlite()
        
        self._initialize_schema()
        
        db_type = "PostgreSQL" if self.is_postgres else f"SQLite ({self.db_path})"
        logger.info(f"✅ MatchDatabase inicializada: {db_type}")

    def _init_postgres(self, database_url: str):
        """Initialize PostgreSQL connection"""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import NullPool
            
            # Fix Railway's postgres:// to postgresql://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            
            logger.info(f"🐘 Connecting to PostgreSQL...")
            
            self.engine = create_engine(
                database_url,
                poolclass=NullPool,  # Railway manages connections
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.is_postgres = True
            logger.info("✅ PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.info("⚠️  Falling back to SQLite...")
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite connection"""
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
            timeout=30.0
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row
        self.is_postgres = False

    def _initialize_schema(self):
        """Crea las tablas si no existen"""
        try:
            schema_path = Path(__file__).parent / "schema_v2.sql"
            
            if not schema_path.exists():
                # Fallback para ruta absoluta (Docker/prod)
                schema_path = Path("/app/src/database/schema_v2.sql")
            
            with open(schema_path, "r") as f:
                schema_script = f.read()
            
            if self.is_postgres:
                # PostgreSQL: Execute using SQLAlchemy
                from sqlalchemy import text
                import re
                
                # Convert SQLite schema to PostgreSQL-compatible
                pg_schema = schema_script
                
                # Replace AUTOINCREMENT with nothing (SERIAL handles it)
                pg_schema = pg_schema.replace("AUTOINCREMENT", "")
                
                # Replace INTEGER PRIMARY KEY with SERIAL PRIMARY KEY
                pg_schema = pg_schema.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
                
                # Fix BOOLEAN defaults (PostgreSQL requires FALSE/TRUE, not 0/1)
                pg_schema = pg_schema.replace("BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE")
                pg_schema = pg_schema.replace("BOOLEAN DEFAULT 1", "BOOLEAN DEFAULT TRUE")
                
                # Fix VIEW syntax: PostgreSQL doesn't support IF NOT EXISTS for CREATE VIEW
                # Use CREATE OR REPLACE VIEW instead
                pg_schema = pg_schema.replace("CREATE VIEW IF NOT EXISTS", "CREATE OR REPLACE VIEW")
                
                # Fix ROUND with FLOAT/REAL: PostgreSQL needs NUMERIC for ROUND with precision
                # The safest way is to remove the entire daily_stats view and recreate it properly
                
                # Remove the problematic daily_stats view definition
                pg_schema = re.sub(
                    r'CREATE OR REPLACE VIEW daily_stats AS.*?ORDER BY fecha DESC;',
                    '',
                    pg_schema,
                    flags=re.DOTALL
                )
                
                # Remove SQLite-specific triggers (PostgreSQL uses different syntax)
                pg_schema = re.sub(r'CREATE TRIGGER.*?END;', '', pg_schema, flags=re.DOTALL)
                
                # Remove the test data insert completely for PostgreSQL
                pg_schema = re.sub(r'-- Insertar partido de ejemplo.*$', '', pg_schema, flags=re.DOTALL)
                
                # Split statements
                raw_statements = pg_schema.split(';')
                
                # Clean and filter statements
                statements = []
                for stmt in raw_statements:
                    # Remove comment lines (lines starting with --)
                    lines = [line for line in stmt.split('\n') if not line.strip().startswith('--')]
                    cleaned = '\n'.join(lines).strip()
                    
                    if cleaned:  # Only keep non-empty statements
                        statements.append(cleaned)
                
                logger.info(f"📊 Processing {len(statements)} SQL statements for PostgreSQL")
                
                # Execute each statement in its own transaction
                for i, statement in enumerate(statements):
                    try:
                        with self.engine.connect() as conn:
                            # Log CREATE TABLE statements for debugging
                            if statement.upper().startswith('CREATE TABLE'):
                                table_name = statement.split()[5] if len(statement.split()) > 5 else "unknown"
                                logger.info(f"✅ Creating table: {table_name}")
                            elif statement.upper().startswith('CREATE OR REPLACE VIEW'):
                                view_name = statement.split()[4] if len(statement.split()) > 4 else "unknown"
                                logger.info(f"✅ Creating view: {view_name}")
                            
                            conn.execute(text(statement))
                            conn.commit()
                    except Exception as e:
                        error_msg = str(e)[:300]
                        if "already exists" in error_msg.lower():
                            logger.debug(f"Table/index already exists (statement {i+1})")
                        else:
                            stmt_preview = statement[:100].replace('\n', ' ')
                            logger.error(f"❌ Statement {i+1} failed: {error_msg}")
                            logger.error(f"   Preview: {stmt_preview}...")
                    
                # Create daily_stats view separately with proper PostgreSQL syntax
                try:
                    daily_stats_view = """
                    CREATE OR REPLACE VIEW daily_stats AS
                    SELECT 
                        DATE(b.timestamp_apuesta) as fecha,
                        COUNT(*) as total_apuestas,
                        SUM(CASE WHEN b.resultado = 'ganada' THEN 1 ELSE 0 END) as ganadas,
                        SUM(CASE WHEN b.resultado = 'perdida' THEN 1 ELSE 0 END) as perdidas,
                        ROUND(
                            (SUM(CASE WHEN b.resultado = 'ganada' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0))::NUMERIC,
                            3
                        ) as win_rate,
                        COALESCE(SUM(b.stake), 0) as stake_total,
                        COALESCE(SUM(b.ganancia), 0) as ganancia_total,
                        ROUND(
                            (COALESCE(SUM(b.ganancia), 0) * 1.0 / NULLIF(SUM(b.stake), 0))::NUMERIC,
                            3
                        ) as roi
                    FROM bets b
                    WHERE b.estado = 'completada'
                    GROUP BY DATE(b.timestamp_apuesta)
                    ORDER BY fecha DESC
                    """
                    with self.engine.connect() as conn:
                        conn.execute(text(daily_stats_view))
                        conn.commit()
                    logger.info("✅ Creating view: daily_stats (PostgreSQL optimized)")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.error(f"Error creating daily_stats view: {e}")
                
                logger.info("✅ PostgreSQL schema initialized")
            else:
                # SQLite: Execute using sqlite3
                self.conn.executescript(schema_script)
                self.conn.commit()
                logger.info("✅ SQLite schema initialized")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando esquema DB: {e}")
            # Don't fail completely, tables might already exist


    # ============================================================
    # DATABASE ABSTRACTION LAYER
    # ============================================================
    
    def _execute(self, query: str, params: tuple = None):
        """Execute a query (works for both SQLite and PostgreSQL)"""
        try:
            if self.is_postgres:
                from sqlalchemy import text
                with self.engine.connect() as conn:
                    result = conn.execute(text(query), params or {})
                    conn.commit()
                    return result
            else:
                cursor = self.conn.cursor()
                cursor.execute(query, params or ())
                self.conn.commit()
                return cursor
        except Exception as e:
            logger.error(f"❌ Error in _execute: {e}")
            logger.error(f"   Query: {query[:200]}...")
            logger.error(f"   Params: {params}")
            raise  # Re-raise para que el llamador pueda manejar
    
    def _fetchone(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Fetch one row (works for both SQLite and PostgreSQL)"""
        try:
            if self.is_postgres:
                from sqlalchemy import text
                with self.engine.connect() as conn:
                    result = conn.execute(text(query), params or {})
                    row = result.fetchone()
                    if row:
                        return dict(row._mapping)
                    return None
            else:
                cursor = self.conn.cursor()
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"❌ Error in _fetchone: {e}")
            logger.error(f"   Query: {query[:200]}...")
            logger.error(f"   Params: {params}")
            return None
    
    def _fetchall(self, query: str, params: tuple = None) -> List[Dict]:
        """Fetch all rows (works for both SQLite and PostgreSQL)"""
        if self.is_postgres:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return [dict(row._mapping) for row in result.fetchall()]
        else:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            return [dict(row) for row in cursor.fetchall()]
    
    def _get_lastrowid(self, result) -> int:
        """Get last inserted row ID (works for both SQLite and PostgreSQL)"""
        if self.is_postgres:
            # For PostgreSQL with SERIAL, we need to use RETURNING id
            # This is handled in the query itself
            return result.fetchone()[0] if result else None
        else:
            return result.lastrowid

    # ============================================================
    # MÉTODOS DE PARTIDOS (MATCHES)
    # ============================================================

    def create_match(
        self,
        fecha_partido: date,
        superficie: str,
        jugador1_nombre: str,
        jugador1_cuota: float,
        jugador2_nombre: str,
        jugador2_cuota: float,
        hora_inicio: Optional[str] = None,
        torneo: Optional[str] = None,
        ronda: Optional[str] = None,
        jugador1_ranking: Optional[int] = None,
        jugador2_ranking: Optional[int] = None,
        # Nuevos campos para tracking
        event_key: Optional[str] = None,
        jugador1_key: Optional[str] = None,
        jugador2_key: Optional[str] = None,
        tournament_key: Optional[str] = None,
        tournament_season: Optional[str] = None,
        event_live: Optional[str] = None,
        event_qualification: Optional[str] = None,
        # Logos de jugadores
        jugador1_logo: Optional[str] = None,
        jugador2_logo: Optional[str] = None,
        # Estado del partido
        estado: Optional[str] = "pendiente",
    ) -> int:
        """
        Crea un nuevo partido

        Returns:
            ID del partido creado
        """
        if self.is_postgres:
            # PostgreSQL: Use RETURNING id
            from sqlalchemy import text
            
            query = """
                INSERT INTO matches (
                    fecha_partido, hora_inicio, torneo, ronda, superficie,
                    jugador1_nombre, jugador1_ranking, jugador1_logo,
                    jugador2_nombre, jugador2_ranking, jugador2_logo,
                    event_key, jugador1_key, jugador2_key,
                    tournament_key, tournament_season,
                    event_live, event_qualification,
                    estado
                ) VALUES (
                    :fecha_partido, :hora_inicio, :torneo, :ronda, :superficie,
                    :jugador1_nombre, :jugador1_ranking, :jugador1_logo,
                    :jugador2_nombre, :jugador2_ranking, :jugador2_logo,
                    :event_key, :jugador1_key, :jugador2_key,
                    :tournament_key, :tournament_season,
                    :event_live, :event_qualification,
                    :estado
                ) RETURNING id
            """
            params = {
                "fecha_partido": fecha_partido,
                "hora_inicio": hora_inicio,
                "torneo": torneo,
                "ronda": ronda,
                "superficie": superficie,
                "jugador1_nombre": jugador1_nombre,
                "jugador1_ranking": jugador1_ranking,
                "jugador1_logo": jugador1_logo,
                "jugador2_nombre": jugador2_nombre,
                "jugador2_ranking": jugador2_ranking,
                "jugador2_logo": jugador2_logo,
                "event_key": event_key,
                "jugador1_key": jugador1_key,
                "jugador2_key": jugador2_key,
                "tournament_key": tournament_key,
                "tournament_season": tournament_season,
                "event_live": event_live,
                "event_qualification": event_qualification,
                "estado": estado,
            }
            
            # CRITICAL FIX: Fetch ID within the connection context
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                match_id = result.fetchone()[0]  # Get ID before connection closes
                conn.commit()
        else:
            # SQLite: Use lastrowid
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO matches (
                    fecha_partido, hora_inicio, torneo, ronda, superficie,
                    jugador1_nombre, jugador1_ranking, jugador1_logo,
                    jugador2_nombre, jugador2_ranking, jugador2_logo,
                    event_key, jugador1_key, jugador2_key,
                    tournament_key, tournament_season,
                    event_live, event_qualification,
                    estado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    fecha_partido,
                    hora_inicio,
                    torneo,
                    ronda,
                    superficie,
                    jugador1_nombre,
                    jugador1_ranking,
                    jugador1_logo,
                    jugador2_nombre,
                    jugador2_ranking,
                    jugador2_logo,
                    event_key,
                    jugador1_key,
                    jugador2_key,
                    tournament_key,
                    tournament_season,
                    event_live,
                    event_qualification,
                    estado,
                ),
            )
            self.conn.commit()
            match_id = cursor.lastrowid

        logger.info(f"✅ Partido creado: {jugador1_nombre} vs {jugador2_nombre} (ID: {match_id})")
        return match_id

    def get_match(self, match_id: int) -> Optional[Dict]:
        """Obtiene un partido por ID"""
        return self._fetchone("SELECT * FROM matches WHERE id = :id", {"id": match_id})

    def get_matches_by_date(self, fecha: date) -> List[Dict]:
        """
        Obtiene todos los partidos de una fecha específica

        Returns:
            Lista de partidos con sus predicciones y resultados
        """
        matches = self._fetchall(
            """
            SELECT * FROM matches_with_latest_prediction
            WHERE fecha_partido = :fecha
            ORDER BY hora_inicio ASC, id ASC
        """,
            {"fecha": fecha},
        )

        logger.info(f"📊 Encontrados {len(matches)} partidos para {fecha}")
        return matches

    def update_match_result(
        self, match_id: int, ganador: str, marcador: Optional[str] = None
    ) -> bool:
        """
        Actualiza el resultado de un partido

        Returns:
            True si se actualizó correctamente
        """
        result = self._execute(
            """
            UPDATE matches
            SET resultado_ganador = :ganador,
                resultado_marcador = :marcador,
                estado = 'completado'
            WHERE id = :match_id
        """,
            {"ganador": ganador, "marcador": marcador, "match_id": match_id},
        )

        if self.is_postgres:
            success = result.rowcount > 0
        else:
            success = result.rowcount > 0

        if success:
            logger.info(f"✅ Resultado actualizado para partido {match_id}: Ganador {ganador}")
            return True

        logger.warning(f"⚠️  No se pudo actualizar resultado del partido {match_id}")
        return False

    def get_live_matches(self) -> List[Dict]:
        """Obtiene partidos que están en vivo"""
        return self._fetchall("""
            SELECT * FROM matches 
            WHERE event_live = '1' 
            AND estado IN ('pendiente', 'en_juego')
            AND event_key IS NOT NULL
        """, {})

    def update_match_live_data(
        self,
        match_id: int,
        scores: str = None,
        event_live: str = None,
        event_status: str = None,
        event_final_result: str = None
    ):
        """Actualiza datos en vivo de un partido"""
        updates = []
        params = {"match_id": match_id}
        
        if scores is not None:
            updates.append("resultado_marcador = :scores")
            params["scores"] = str(scores)
        
        if event_live is not None:
            updates.append("event_live = :event_live")
            params["event_live"] = event_live
        
        if event_status is not None:
            # Mapear estado de API a nuestro estado (incl. Retired, Walk Over, etc.)
            event_status_lower = event_status.lower()
            finished_keywords = [
                "finished", "ended", "completed", "final",
                "walk over", "walkover", "w.o.", "wo", "w/o",
                "retired", "ret", "retirement",
                "defaulted", "def", "default", "awarded",
                "cancelled", "canceled", "postponed", "suspended", "interrupted",
            ]
            if any(kw in event_status_lower for kw in finished_keywords):
                updates.append("estado = 'completado'")
            elif event_live == "1":
                updates.append("estado = 'en_juego'")
        
        if event_final_result is not None:
            updates.append("event_final_result = :event_final_result")
            params["event_final_result"] = event_final_result
        
        if updates:
            query = f"UPDATE matches SET {', '.join(updates)} WHERE id = :match_id"
            self._execute(query, params)
            logger.debug(f"Actualizado partido en vivo {match_id}")

    def get_match_by_event_key(self, event_key: str) -> Optional[Dict]:
        """
        Obtiene un partido por su event_key de API-Tennis
        
        Args:
            event_key: ID del partido en API-Tennis
        
        Returns:
            Dict con datos del partido o None si no se encuentra
        """
        return self._fetchone("SELECT * FROM matches WHERE event_key = :event_key", {"event_key": event_key})

    def get_recent_matches(self, days: int = 7) -> List[Dict]:
        """
        Obtiene partidos de los últimos N días
        
        Args:
            days: Número de días hacia atrás (default: 7)
        
        Returns:
            Lista de partidos
        """
        from datetime import date, timedelta
        
        fecha_inicio = date.today() - timedelta(days=days)
        
        return self._fetchall(
            """
            SELECT * FROM matches 
            WHERE fecha_partido >= :fecha_inicio
            ORDER BY fecha_partido DESC, hora_inicio DESC
            """,
            {"fecha_inicio": str(fecha_inicio)},
        )

    def delete_match(self, match_id: int) -> bool:
        """
        Elimina un partido y todas sus predicciones/apuestas asociadas

        Returns:
            True si se eliminó correctamente
        """
        try:
            # Las foreign keys con ON DELETE CASCADE se encargan de eliminar
            # predicciones y apuestas automáticamente
            self._execute("DELETE FROM bets WHERE match_id = :match_id", {"match_id": match_id})
            logger.debug(f"Eliminadas apuestas del partido {match_id}")

            self._execute("DELETE FROM predictions WHERE match_id = :match_id", {"match_id": match_id})
            logger.debug(f"Eliminadas predicciones del partido {match_id}")

            self._execute("DELETE FROM matches WHERE id = :match_id", {"match_id": match_id})
            logger.debug(f"Eliminado partido {match_id}")

            logger.info(f"✅ Partido {match_id} eliminado completamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error eliminando partido {match_id}: {e}")
            return False

    def match_exists(
        self, jugador1_nombre: str, jugador2_nombre: str, fecha_partido: date
    ) -> bool:
        """
        Verifica si ya existe un partido entre dos jugadores en una fecha

        Args:
            jugador1_nombre: Nombre del primer jugador
            jugador2_nombre: Nombre del segundo jugador
            fecha_partido: Fecha del partido

        Returns:
            True si el partido ya existe
        """
        # DEBUG: Log the query parameters
        logger.info(f"🔍 match_exists() called: {jugador1_nombre} vs {jugador2_nombre} on {fecha_partido}")
        
        # Buscar en ambas direcciones (J1 vs J2 o J2 vs J1)
        result = self._fetchone(
            """
            SELECT id FROM matches
            WHERE fecha_partido = :fecha
            AND (
                (jugador1_nombre = :j1 AND jugador2_nombre = :j2)
                OR
                (jugador1_nombre = :j2 AND jugador2_nombre = :j1)
            )
        """,
            {
                "fecha": fecha_partido,
                "j1": jugador1_nombre,
                "j2": jugador2_nombre,
            },
        )

        exists = result is not None
        
        # DEBUG: Log the result
        if exists:
            logger.info(f"✅ Partido EXISTE en DB: {jugador1_nombre} vs {jugador2_nombre} (ID: {result.get('id')})")
        else:
            logger.info(f"❌ Partido NO existe en DB: {jugador1_nombre} vs {jugador2_nombre}")

        return exists

    def get_matches_date_range(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Obtiene todos los partidos en un rango de fechas

        Args:
            start_date: Fecha inicial (inclusive)
            end_date: Fecha final (inclusive)

        Returns:
            Lista de partidos con predicciones
        """
        matches = self._fetchall(
            """
            SELECT * FROM matches_with_latest_prediction
            WHERE fecha_partido BETWEEN :start_date AND :end_date
            ORDER BY fecha_partido ASC, hora_inicio ASC, id ASC
        """,
            {"start_date": start_date, "end_date": end_date},
        )

        logger.info(
            f"📊 Encontrados {len(matches)} partidos entre {start_date} y {end_date}"
        )
        return matches

    def cleanup_old_matches(self, days_to_keep: int = 30) -> int:
        """
        Elimina partidos antiguos (más de X días)

        Args:
            days_to_keep: Días de historial a mantener (default: 30)

        Returns:
            Número de partidos eliminados
        """
        from datetime import date, timedelta

        cutoff_date = date.today() - timedelta(days=days_to_keep)

        # Contar partidos a eliminar
        count_result = self._fetchone(
            """
            SELECT COUNT(*) as count FROM matches
            WHERE fecha_partido < :cutoff_date
        """,
            {"cutoff_date": str(cutoff_date)},
        )

        count = count_result["count"] if count_result else 0

        if count > 0:
            # Eliminar partidos antiguos
            self._execute(
                "DELETE FROM matches WHERE fecha_partido < :cutoff_date",
                {"cutoff_date": str(cutoff_date)},
            )

            logger.info(f"🗑️  Eliminados {count} partidos anteriores a {cutoff_date}")
            return count
        else:
            logger.info(f"✅ No hay partidos anteriores a {cutoff_date} para eliminar")
            return 0


    # ============================================================
    # MÉTODOS DE PREDICCIONES
    # ============================================================

    @staticmethod
    def _to_python_type(value):
        """Convierte tipos numpy a tipos nativos de Python para compatibilidad con PostgreSQL"""
        if value is None:
            return None
        # Convertir numpy types a Python types
        if hasattr(value, 'item'):  # numpy scalar
            return value.item()
        return value

    def add_prediction(
        self,
        match_id: int,
        jugador1_cuota: float,
        jugador2_cuota: float,
        jugador1_probabilidad: float,
        jugador2_probabilidad: float,
        jugador1_ev: float,
        jugador2_ev: float,
        recomendacion: str,
        mejor_opcion: Optional[str] = None,
        confianza: Optional[str] = None,
        jugador1_edge: Optional[float] = None,
        jugador2_edge: Optional[float] = None,
        kelly_stake_jugador1: Optional[float] = None,
        kelly_stake_jugador2: Optional[float] = None,
        # Nuevos parámetros de confianza
        confidence_level: Optional[str] = None,
        confidence_score: Optional[float] = None,
        player1_known: Optional[bool] = None,
        player2_known: Optional[bool] = None,
    ) -> int:
        """
        Añade una nueva versión de predicción para un partido

        Returns:
            ID de la predicción creada
        """
        # Convertir tipos numpy a tipos nativos de Python
        jugador1_cuota = self._to_python_type(jugador1_cuota)
        jugador2_cuota = self._to_python_type(jugador2_cuota)
        jugador1_probabilidad = self._to_python_type(jugador1_probabilidad)
        jugador2_probabilidad = self._to_python_type(jugador2_probabilidad)
        jugador1_ev = self._to_python_type(jugador1_ev)
        jugador2_ev = self._to_python_type(jugador2_ev)
        jugador1_edge = self._to_python_type(jugador1_edge)
        jugador2_edge = self._to_python_type(jugador2_edge)
        kelly_stake_jugador1 = self._to_python_type(kelly_stake_jugador1)
        kelly_stake_jugador2 = self._to_python_type(kelly_stake_jugador2)
        confidence_score = self._to_python_type(confidence_score)
        # Obtener la última versión
        max_version_row = self._fetchone(
            """
            SELECT COALESCE(MAX(version), 0) as max_version
            FROM predictions
            WHERE match_id = :match_id
        """,
            {"match_id": match_id},
        )
        
        max_version = max_version_row["max_version"] if max_version_row else 0
        new_version = max_version + 1

        # Insertar nueva predicción
        if self.is_postgres:
            query = """
                INSERT INTO predictions (
                    match_id, version,
                    jugador1_cuota, jugador2_cuota,
                    jugador1_probabilidad, jugador2_probabilidad,
                    jugador1_ev, jugador2_ev,
                    jugador1_edge, jugador2_edge,
                    recomendacion, mejor_opcion, confianza,
                    kelly_stake_jugador1, kelly_stake_jugador2,
                    confidence_level, confidence_score, player1_known, player2_known
                ) VALUES (
                    :match_id, :version,
                    :jugador1_cuota, :jugador2_cuota,
                    :jugador1_probabilidad, :jugador2_probabilidad,
                    :jugador1_ev, :jugador2_ev,
                    :jugador1_edge, :jugador2_edge,
                    :recomendacion, :mejor_opcion, :confianza,
                    :kelly_stake_jugador1, :kelly_stake_jugador2,
                    :confidence_level, :confidence_score, :player1_known, :player2_known
                ) RETURNING id
            """
            params = {
                "match_id": match_id,
                "version": new_version,
                "jugador1_cuota": jugador1_cuota,
                "jugador2_cuota": jugador2_cuota,
                "jugador1_probabilidad": jugador1_probabilidad,
                "jugador2_probabilidad": jugador2_probabilidad,
                "jugador1_ev": jugador1_ev,
                "jugador2_ev": jugador2_ev,
                "jugador1_edge": jugador1_edge,
                "jugador2_edge": jugador2_edge,
                "recomendacion": recomendacion,
                "mejor_opcion": mejor_opcion,
                "confianza": confianza,
                "kelly_stake_jugador1": kelly_stake_jugador1,
                "kelly_stake_jugador2": kelly_stake_jugador2,
                "confidence_level": confidence_level,
                "confidence_score": confidence_score,
                "player1_known": player1_known,
                "player2_known": player2_known,
            }
            result = self._execute(query, params)
            prediction_id = result.fetchone()[0]
        else:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (
                    match_id, version,
                    jugador1_cuota, jugador2_cuota,
                    jugador1_probabilidad, jugador2_probabilidad,
                    jugador1_ev, jugador2_ev,
                    jugador1_edge, jugador2_edge,
                    recomendacion, mejor_opcion, confianza,
                    kelly_stake_jugador1, kelly_stake_jugador2,
                    confidence_level, confidence_score, player1_known, player2_known
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    match_id,
                    new_version,
                    jugador1_cuota,
                    jugador2_cuota,
                    jugador1_probabilidad,
                    jugador2_probabilidad,
                    jugador1_ev,
                    jugador2_ev,
                    jugador1_edge,
                    jugador2_edge,
                    recomendacion,
                    mejor_opcion,
                    confianza,
                    kelly_stake_jugador1,
                    kelly_stake_jugador2,
                    confidence_level,
                    confidence_score,
                    1 if player1_known else 0,
                    1 if player2_known else 0,
                ),
            )
            self.conn.commit()
            prediction_id = cursor.lastrowid
        
        # Log con información de confianza
        confidence_msg = f" (Confianza: {confidence_level})" if confidence_level else ""
        logger.info(f"✅ Predicción v{new_version} creada para partido {match_id}: {recomendacion}{confidence_msg}")
        return prediction_id

    def get_latest_prediction(self, match_id: int) -> Optional[Dict]:
        """Obtiene la última predicción de un partido"""
        return self._fetchone(
            """
            SELECT * FROM predictions
            WHERE match_id = :match_id
            ORDER BY version DESC
            LIMIT 1
        """,
            {"match_id": match_id},
        )

    def get_prediction_history(self, match_id: int) -> List[Dict]:
        """Obtiene todas las versiones de predicción de un partido"""
        return self._fetchall(
            """
            SELECT * FROM predictions
            WHERE match_id = :match_id
            ORDER BY version ASC
        """,
            {"match_id": match_id},
        )

    # ============================================================
    # MÉTODOS DE APUESTAS (BETS)
    # ============================================================

    def register_bet(
        self,
        match_id: int,
        prediction_id: int,
        jugador_apostado: str,
        cuota_apostada: float,
        stake: float,
    ) -> int:
        """
        Registra una nueva apuesta

        Returns:
            ID de la apuesta creada
        """
        # Verificar si ya existe una apuesta activa para este partido
        existing_bet = self._fetchone(
            """
            SELECT id FROM bets
            WHERE match_id = :match_id AND estado = 'activa'
        """,
            {"match_id": match_id},
        )

        if existing_bet:
            logger.warning(f"⚠️  Ya existe una apuesta activa para el partido {match_id}")
            return existing_bet["id"]

        # Crear nueva apuesta
        if self.is_postgres:
            query = """
                INSERT INTO bets (
                    match_id, prediction_id,
                    jugador_apostado, cuota_apostada, stake,
                    estado
                ) VALUES (
                    :match_id, :prediction_id,
                    :jugador_apostado, :cuota_apostada, :stake,
                    'activa'
                ) RETURNING id
            """
            params = {
                "match_id": match_id,
                "prediction_id": prediction_id,
                "jugador_apostado": jugador_apostado,
                "cuota_apostada": cuota_apostada,
                "stake": stake,
            }
            result = self._execute(query, params)
            bet_id = result.fetchone()[0]
        else:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO bets (
                    match_id, prediction_id,
                    jugador_apostado, cuota_apostada, stake,
                    estado
                ) VALUES (?, ?, ?, ?, ?, 'activa')
            """,
                (match_id, prediction_id, jugador_apostado, cuota_apostada, stake),
            )
            self.conn.commit()
            bet_id = cursor.lastrowid

        logger.info(
            f"✅ Apuesta registrada: {jugador_apostado} @ {cuota_apostada} (Stake: {stake}€)"
        )
        return bet_id

    def update_bet_result(self, match_id: int, ganador: str) -> bool:
        """
        Actualiza el resultado de una apuesta cuando el partido termina

        Returns:
            True si se actualizó correctamente
        """
        # Obtener la apuesta activa
        bet = self._fetchone(
            """
            SELECT * FROM bets
            WHERE match_id = :match_id AND estado = 'activa'
        """,
            {"match_id": match_id},
        )

        if not bet:
            logger.warning(f"⚠️  No hay apuesta activa para el partido {match_id}")
            return False

        # Calcular resultado
        jugador_apostado = bet["jugador_apostado"]
        cuota = bet["cuota_apostada"]
        stake = bet["stake"]

        if jugador_apostado == ganador:
            # Apuesta ganada
            resultado = "ganada"
            ganancia = stake * (cuota - 1)  # Ganancia neta
            roi = cuota - 1
        else:
            # Apuesta perdida
            resultado = "perdida"
            ganancia = -stake
            roi = -1.0

        # Actualizar apuesta
        self._execute(
            """
            UPDATE bets
            SET resultado = :resultado,
                ganancia = :ganancia,
                roi = :roi,
                estado = 'completada'
            WHERE id = :bet_id
        """,
            {
                "resultado": resultado,
                "ganancia": ganancia,
                "roi": roi,
                "bet_id": bet["id"],
            },
        )

        logger.info(f"✅ Apuesta actualizada: {resultado.upper()} (Ganancia: {ganancia:+.2f}€)")
        return True

    # ============================================================
    # MÉTODOS DE ESTADÍSTICAS
    # ============================================================

    def get_stats_summary(self, days: Optional[int] = 7) -> Dict:
        """
        Calcula estadísticas de rendimiento

        Args:
            days: Número de días hacia atrás (None = todos)

        Returns:
            Dict con estadísticas completas
        """
        # Filtro de fecha
        if days:
            from datetime import timedelta
            fecha_inicio = date.today() - timedelta(days=days)
            date_filter = "AND DATE(b.timestamp_apuesta) >= :fecha_inicio"
            params = {"fecha_inicio": str(fecha_inicio)}
        else:
            fecha_inicio = None
            date_filter = ""
            params = {}

        # Estadísticas de apuestas
        stats_row = self._fetchone(
            f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN resultado = 'ganada' THEN 1 ELSE 0 END) as ganadas,
                SUM(CASE WHEN resultado = 'perdida' THEN 1 ELSE 0 END) as perdidas,
                SUM(stake) as stake_total,
                SUM(ganancia) as ganancia_neta
            FROM bets b
            WHERE estado = 'completada'
            {date_filter}
        """,
            params,
        )

        stats = stats_row if stats_row else {}

        # Calcular métricas derivadas
        total = stats.get("total", 0) or 0
        ganadas = stats.get("ganadas", 0) or 0
        stake_total = stats.get("stake_total", 0) or 0
        ganancia_neta = stats.get("ganancia_neta", 0) or 0

        win_rate = ganadas / total if total > 0 else 0.0
        roi = ganancia_neta / stake_total if stake_total > 0 else 0.0
        ganancia_bruta = ganancia_neta + stake_total if ganancia_neta >= 0 else 0

        return {
            "periodo": f"Últimos {days} días" if days else "Todos los tiempos",
            "fecha_inicio": fecha_inicio,
            "fecha_fin": date.today(),
            "apuestas": {
                "total": total,
                "ganadas": ganadas,
                "perdidas": stats.get("perdidas", 0) or 0,
                "win_rate": round(win_rate, 3),
            },
            "financiero": {
                "stake_total": round(stake_total, 2),
                "ganancia_bruta": round(ganancia_bruta, 2),
                "ganancia_neta": round(ganancia_neta, 2),
                "roi": round(roi, 3),
            },
            "modelo": {
                "accuracy": None,  # TODO: Calcular desde predictions
                "brier_score": None,
                "ev_promedio": None,
            },
        }

    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """
        Obtiene estadísticas diarias para gráficos

        Returns:
            Lista de estadísticas por día
        """
        from datetime import timedelta
        fecha_inicio = date.today() - timedelta(days=days)

        return self._fetchall(
            """
            SELECT * FROM daily_stats
            WHERE fecha >= :fecha_inicio
            ORDER BY fecha DESC
        """,
            {"fecha_inicio": str(fecha_inicio)},
        )

    # ============================================================
    # MÉTODOS DE CUOTAS (ODDS HISTORY)
    # ============================================================

    def save_top3_odds(
        self,
        match_id: int,
        top3_player1: List[Dict],
        top3_player2: List[Dict],
    ) -> bool:
        """
        Guarda el top 3 de cuotas para cada jugador
        
        Args:
            match_id: ID del partido
            top3_player1: Lista de dicts con bookmaker, odds, is_best
            top3_player2: Lista de dicts con bookmaker, odds, is_best
            
        Returns:
            True si se guardó correctamente
        """
        try:
            # Guardar cuotas de jugador 1
            for odds_info in top3_player1:
                self._execute(
                    """
                    INSERT INTO odds_history (
                        match_id, jugador1_cuota, jugador2_cuota,
                        bookmaker, is_best
                    ) VALUES (:match_id, :j1_cuota, :j2_cuota, :bookmaker, :is_best)
                    """,
                    {
                        "match_id": match_id,
                        "j1_cuota": odds_info["odds"],
                        "j2_cuota": 0.0,  # Placeholder para jugador2
                        "bookmaker": odds_info["bookmaker"],
                        "is_best": odds_info["is_best"],
                    },
                )
            
            # Guardar cuotas de jugador 2
            for odds_info in top3_player2:
                self._execute(
                    """
                    INSERT INTO odds_history (
                        match_id, jugador1_cuota, jugador2_cuota,
                        bookmaker, is_best
                    ) VALUES (:match_id, :j1_cuota, :j2_cuota, :bookmaker, :is_best)
                    """,
                    {
                        "match_id": match_id,
                        "j1_cuota": 0.0,  # Placeholder para jugador1
                        "j2_cuota": odds_info["odds"],
                        "bookmaker": odds_info["bookmaker"],
                        "is_best": odds_info["is_best"],
                    },
                )
            
            logger.info(f"✅ Top 3 cuotas guardadas para partido {match_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando top 3 cuotas: {e}")
            return False

    def get_top3_odds(self, match_id: int) -> Dict:
        """
        Obtiene el top 3 de cuotas para un partido
        
        Args:
            match_id: ID del partido
            
        Returns:
            Dict con top3_player1 y top3_player2
        """
        # Obtener cuotas de jugador 1 (donde jugador1_cuota > 0)
        rows_p1 = self._fetchall(
            """
            SELECT bookmaker, jugador1_cuota as odds, is_best
            FROM odds_history
            WHERE match_id = :match_id AND jugador1_cuota > 0
            ORDER BY jugador1_cuota DESC
            LIMIT 3
            """,
            {"match_id": match_id},
        )
        
        top3_player1 = [
            {
                "bookmaker": row["bookmaker"],
                "odds": row["odds"],
                "is_best": bool(row["is_best"]),
            }
            for row in rows_p1
        ]
        
        # Obtener cuotas de jugador 2 (donde jugador2_cuota > 0)
        rows_p2 = self._fetchall(
            """
            SELECT bookmaker, jugador2_cuota as odds, is_best
            FROM odds_history
            WHERE match_id = :match_id AND jugador2_cuota > 0
            ORDER BY jugador2_cuota DESC
            LIMIT 3
            """,
            {"match_id": match_id},
        )
        
        top3_player2 = [
            {
                "bookmaker": row["bookmaker"],
                "odds": row["odds"],
                "is_best": bool(row["is_best"]),
            }
            for row in rows_p2
        ]
        
        return {
            "top3_player1": top3_player1,
            "top3_player2": top3_player2,
        }

    # ============================================================
    # MÉTODOS DE UTILIDAD
    # ============================================================

    def close(self):
        """Cierra la conexión a la base de datos"""
        if self.conn:
            self.conn.close()
            logger.info("🔒 Conexión a base de datos cerrada")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ============================================================
# EJEMPLO DE USO
# ============================================================


    # ============================================================
    # HELPER METHODS FOR SERVICES (PostgreSQL compatible)
    # ============================================================

    def update_match_hora_inicio(self, match_id: int, hora_inicio: str) -> bool:
        """
        Actualiza la hora de inicio de un partido
        
        Args:
            match_id: ID del partido
            hora_inicio: Nueva hora de inicio
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            self._execute(
                "UPDATE matches SET hora_inicio = :hora WHERE id = :match_id",
                {"hora": hora_inicio, "match_id": match_id}
            )
            return True
        except Exception as e:
            logger.error(f"Error actualizando hora de inicio: {e}")
            return False

    def update_match_ganador(self, match_id: int, ganador: str) -> bool:
        """
        Actualiza el ganador de un partido
        
        Args:
            match_id: ID del partido
            ganador: Nombre del ganador
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            self._execute(
                "UPDATE matches SET resultado_ganador = :ganador WHERE id = :match_id",
                {"ganador": ganador, "match_id": match_id}
            )
            return True
        except Exception as e:
            logger.error(f"Error actualizando ganador: {e}")
            return False

    def update_match_player_keys(self, match_id: int, player1_key: str, player2_key: str) -> bool:
        """
        Actualiza los player_keys de un partido
        
        Args:
            match_id: ID del partido
            player1_key: Key del jugador 1
            player2_key: Key del jugador 2
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            self._execute(
                """
                UPDATE matches
                SET jugador1_key = :player1_key, jugador2_key = :player2_key
                WHERE id = :match_id
            """,
                {"player1_key": player1_key, "player2_key": player2_key, "match_id": match_id}
            )
            return True
        except Exception as e:
            logger.error(f"Error actualizando player keys: {e}")
            return False

    def check_match_games_exist(self, match_id: int) -> int:
        """
        Verifica si ya existen juegos guardados para un partido
        
        Args:
            match_id: ID del partido
            
        Returns:
            Número de juegos existentes
        """
        result = self._fetchone(
            "SELECT COUNT(*) as count FROM match_games WHERE match_id = :match_id",
            {"match_id": match_id}
        )
        return result["count"] if result else 0

    def get_match_sets(self, match_id: int) -> List[Dict]:
        """
        Obtiene los scores por set de un partido
        
        Args:
            match_id: ID del partido
            
        Returns:
            Lista de diccionarios con scores por set
        """
        return self._fetchall(
            """
            SELECT set_number, player1_score, player2_score, tiebreak_score
            FROM match_sets
            WHERE match_id = :match_id
            ORDER BY set_number ASC
            """,
            {"match_id": match_id}
        )

    def save_match_sets(self, match_id: int, sets_data: List[Dict]) -> int:
        """
        Guarda los scores por set de un partido
        
        Args:
            match_id: ID del partido
            sets_data: Lista de diccionarios con datos de cada set
                       [{"set_number": 1, "player1_score": 6, "player2_score": 4}, ...]
            
        Returns:
            Número de sets guardados
        """
        saved = 0
        for set_data in sets_data:
            try:
                # Usar UPSERT para no duplicar
                if self.is_postgres:
                    query = """
                        INSERT INTO match_sets (match_id, set_number, player1_score, player2_score, tiebreak_score)
                        VALUES (:match_id, :set_number, :player1_score, :player2_score, :tiebreak_score)
                        ON CONFLICT (match_id, set_number) 
                        DO UPDATE SET player1_score = :player1_score, player2_score = :player2_score, 
                                      tiebreak_score = :tiebreak_score
                    """
                else:
                    query = """
                        INSERT OR REPLACE INTO match_sets (match_id, set_number, player1_score, player2_score, tiebreak_score)
                        VALUES (:match_id, :set_number, :player1_score, :player2_score, :tiebreak_score)
                    """
                
                self._execute(query, {
                    "match_id": match_id,
                    "set_number": set_data.get("set_number", 0),
                    "player1_score": set_data.get("player1_score", 0),
                    "player2_score": set_data.get("player2_score", 0),
                    "tiebreak_score": set_data.get("tiebreak_score")
                })
                saved += 1
            except Exception as e:
                logger.debug(f"Error guardando set: {e}")
        
        return saved

    def save_match_game(self, match_id: int, game_data: Dict) -> bool:
        """
        Guarda un juego de un partido
        
        Args:
            match_id: ID del partido
            game_data: Datos del juego
            
        Returns:
            True si se guardó correctamente
        """
        try:
            self._execute(
                """
                INSERT INTO match_games (
                    match_id, set_number, game_number,
                    server, winner, score_games, score_sets, was_break
                ) VALUES (
                    :match_id, :set_number, :game_number,
                    :server, :winner, :score_games, :score_sets, :was_break
                )
            """,
                {
                    "match_id": match_id,
                    "set_number": game_data.get("set_number", ""),
                    "game_number": game_data.get("number_game", ""),
                    "server": game_data.get("player_served", ""),
                    "winner": game_data.get("serve_winner", ""),
                    "score_games": game_data.get("score", ""),
                    "score_sets": "",
                    "was_break": 1 if game_data.get("serve_lost") else 0
                }
            )
            return True
        except Exception as e:
            logger.debug(f"Error guardando juego: {e}")
            return False

    def save_match_point(self, match_id: int, set_number: str, game_number: int, point_data: Dict) -> bool:
        """
        Guarda un punto de un partido
        
        Args:
            match_id: ID del partido
            set_number: Número de set
            game_number: Número de juego
            point_data: Datos del punto
            
        Returns:
            True si se guardó correctamente
        """
        try:
            self._execute(
                """
                INSERT INTO match_pointbypoint (
                    match_id, set_number, game_number, point_number,
                    server, score, is_break_point, is_set_point, is_match_point
                ) VALUES (
                    :match_id, :set_number, :game_number, :point_number,
                    :server, :score, :is_break_point, :is_set_point, :is_match_point
                )
            """,
                {
                    "match_id": match_id,
                    "set_number": set_number,
                    "game_number": game_number,
                    "point_number": point_data.get("number_point", ""),
                    "server": point_data.get("player_served", ""),
                    "score": point_data.get("score", ""),
                    "is_break_point": 1 if point_data.get("break_point") else 0,
                    "is_set_point": 1 if point_data.get("set_point") else 0,
                    "is_match_point": 1 if point_data.get("match_point") else 0
                }
            )
            return True
        except Exception as e:
            logger.debug(f"Error guardando punto: {e}")
            return False


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    # Eliminar base de datos de prueba si existe
    test_db = "test_matches.db"
    if os.path.exists(test_db):
        os.remove(test_db)
        print("🗑️  Base de datos de prueba eliminada")

    # Crear instancia de database
    db = MatchDatabase(test_db)

    # Crear un partido
    match_id = db.create_match(
        fecha_partido=date.today(),
        superficie="Hard",
        jugador1_nombre="Djokovic",
        jugador1_cuota=1.85,
        jugador2_nombre="Nadal",
        jugador2_cuota=2.10,
        torneo="Australian Open",
        ronda="Semifinal",
    )

    # Añadir predicción
    prediction_id = db.add_prediction(
        match_id=match_id,
        jugador1_cuota=1.85,
        jugador2_cuota=2.10,
        jugador1_probabilidad=0.58,
        jugador2_probabilidad=0.42,
        jugador1_ev=0.073,
        jugador2_ev=-0.118,
        recomendacion="APOSTAR a Djokovic",
        mejor_opcion="Djokovic",
        confianza="Alta",
        jugador1_edge=0.04,
        jugador2_edge=-0.06,
        kelly_stake_jugador1=12.5,
    )

    # Registrar apuesta
    bet_id = db.register_bet(
        match_id=match_id,
        prediction_id=prediction_id,
        jugador_apostado="Djokovic",
        cuota_apostada=1.85,
        stake=12.5,
    )

    # Obtener partidos de hoy
    partidos_hoy = db.get_matches_by_date(date.today())
    print(f"\n📊 Partidos de hoy: {len(partidos_hoy)}")
    for partido in partidos_hoy:
        print(f"  - {partido['jugador1_nombre']} vs {partido['jugador2_nombre']}")
        print(f"    Recomendación: {partido['recomendacion']}")

    # Actualizar resultado del partido
    db.update_match_result(match_id, ganador="Djokovic", marcador="6-4, 7-5")
    db.update_bet_result(match_id, ganador="Djokovic")

    # Obtener estadísticas
    stats = db.get_stats_summary(days=7)
    print(f"\n📈 Estadísticas (7 días):")
    print(f"  Total apuestas: {stats['apuestas']['total']}")
    print(f"  Ganadas: {stats['apuestas']['ganadas']}")
    print(f"  Win rate: {stats['apuestas']['win_rate']*100:.1f}%")
    print(f"  ROI: {stats['financiero']['roi']*100:.1f}%")
    print(f"  Ganancia neta: {stats['financiero']['ganancia_neta']:+.2f}€")

    db.close()
    print("\n✅ Test completado exitosamente!")
