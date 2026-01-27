"""
Sistema de Base de Datos para Tracking de Predicciones
Fase 4: Tennis ML Predictor

Soporta SQLite (desarrollo local) y PostgreSQL (producción Railway).
Detecta automáticamente DATABASE_URL para usar PostgreSQL.
"""

import sqlite3
import os
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TennisDatabase:
    """
    Sistema de base de datos para tracking de predicciones y apuestas.
    
    Soporta SQLite (local) y PostgreSQL (Railway production).
    Detecta automáticamente DATABASE_URL y usa PostgreSQL si está disponible.
    """

    def __init__(self, db_path="apuestas_tracker.db"):
        """
        Inicializa la base de datos

        Args:
            db_path: Ruta al archivo de base de datos SQLite (usado solo si no hay DATABASE_URL)
        """
        self.db_path = db_path
        self.conn = None
        self.is_postgres = False
        self.engine = None

        # Check for PostgreSQL URL (Railway)
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            self._init_postgres(database_url)
        else:
            self._init_sqlite()

        # Crear DB si no existe
        self.crear_database()
        
        db_type = "PostgreSQL" if self.is_postgres else f"SQLite ({self.db_path})"
        logger.info(f"✅ TennisDatabase inicializada: {db_type}")

    def _init_postgres(self, database_url: str):
        """Initialize PostgreSQL connection"""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import NullPool
            
            # Fix Railway's postgres:// to postgresql://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            
            logger.info("🐘 Connecting TennisDatabase to PostgreSQL...")
            
            self.engine = create_engine(
                database_url,
                poolclass=NullPool,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.is_postgres = True
            logger.info("✅ TennisDatabase PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            logger.info("⚠️  Falling back to SQLite...")
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite connection"""
        self.is_postgres = False
        # Connection will be created on demand

    def conectar(self):
        """Conectar a la base de datos (SQLite mode)"""
        if self.is_postgres:
            return None  # PostgreSQL uses engine
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def desconectar(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _execute(self, query: str, params: dict = None):
        """Execute a query (works for both SQLite and PostgreSQL)"""
        if self.is_postgres:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                conn.commit()
                return result
        else:
            conn = self.conectar()
            cursor = conn.cursor()
            # Convert :param to ? for SQLite
            if params:
                sqlite_query = query
                values = []
                for key, value in params.items():
                    sqlite_query = sqlite_query.replace(f":{key}", "?")
                    values.append(value)
                cursor.execute(sqlite_query, values)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor

    def _fetchone(self, query: str, params: dict = None):
        """Fetch one row"""
        if self.is_postgres:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None
        else:
            conn = self.conectar()
            cursor = conn.cursor()
            if params:
                sqlite_query = query
                values = []
                for key, value in params.items():
                    sqlite_query = sqlite_query.replace(f":{key}", "?")
                    values.append(value)
                cursor.execute(sqlite_query, values)
            else:
                cursor.execute(query)
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def _fetchall(self, query: str, params: dict = None):
        """Fetch all rows"""
        if self.is_postgres:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return [dict(row._mapping) for row in result.fetchall()]
        else:
            conn = self.conectar()
            cursor = conn.cursor()
            if params:
                sqlite_query = query
                values = []
                for key, value in params.items():
                    sqlite_query = sqlite_query.replace(f":{key}", "?")
                    values.append(value)
                cursor.execute(sqlite_query, values)
            else:
                cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def crear_database(self):
        """
        Crea todas las tablas necesarias.
        Soporta SQLite y PostgreSQL.
        """
        if self.is_postgres:
            self._crear_database_postgres()
        else:
            self._crear_database_sqlite()

    def _crear_database_postgres(self):
        """Crea tablas en PostgreSQL"""
        from sqlalchemy import text
        
        statements = [
            # Tabla: Predicciones
            """
            CREATE TABLE IF NOT EXISTS predicciones (
                id SERIAL PRIMARY KEY,
                fecha_partido DATE NOT NULL,
                fecha_prediccion TIMESTAMP NOT NULL,
                
                jugador_nombre TEXT NOT NULL,
                jugador_rank INTEGER,
                oponente_nombre TEXT NOT NULL,
                oponente_rank INTEGER,
                superficie TEXT,
                torneo TEXT,
                ronda TEXT,
                
                prob_modelo REAL NOT NULL,
                prob_modelo_calibrada REAL,
                
                cuota REAL NOT NULL,
                bookmaker TEXT,
                ev REAL NOT NULL,
                umbral_ev REAL DEFAULT 0.03,
                
                decision TEXT NOT NULL,
                apuesta_cantidad REAL,
                
                resultado_real INTEGER,
                ganancia REAL,
                fecha_resultado TIMESTAMP,
                
                modelo_usado TEXT,
                version_modelo TEXT,
                notas TEXT
            )
            """,
            # Tabla: Configuración
            """
            CREATE TABLE IF NOT EXISTS configuracion (
                id SERIAL PRIMARY KEY,
                parametro TEXT UNIQUE NOT NULL,
                valor TEXT NOT NULL,
                fecha_modificacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Tabla: Estadísticas diarias
            """
            CREATE TABLE IF NOT EXISTS estadisticas_diarias (
                id SERIAL PRIMARY KEY,
                fecha DATE UNIQUE NOT NULL,
                predicciones_totales INTEGER DEFAULT 0,
                apuestas_realizadas INTEGER DEFAULT 0,
                apuestas_ganadas INTEGER DEFAULT 0,
                apuestas_perdidas INTEGER DEFAULT 0,
                apuestas_pendientes INTEGER DEFAULT 0,
                total_apostado REAL DEFAULT 0,
                total_ganado REAL DEFAULT 0,
                roi REAL,
                ev_promedio REAL,
                win_rate REAL
            )
            """,
            # Índices
            "CREATE INDEX IF NOT EXISTS idx_pred_fecha_partido ON predicciones(fecha_partido)",
            "CREATE INDEX IF NOT EXISTS idx_pred_decision ON predicciones(decision)",
            "CREATE INDEX IF NOT EXISTS idx_pred_resultado ON predicciones(resultado_real)",
        ]
        
        for stmt in statements:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(stmt))
                    conn.commit()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Error creando tabla: {e}")
        
        logger.info("✅ TennisDatabase PostgreSQL schema initialized")

    def _crear_database_sqlite(self):
        """Crea tablas en SQLite"""
        conn = self.conectar()
        cursor = conn.cursor()

        # Tabla: Predicciones
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_partido DATE NOT NULL,
                fecha_prediccion DATETIME NOT NULL,
                
                jugador_nombre TEXT NOT NULL,
                jugador_rank INTEGER,
                oponente_nombre TEXT NOT NULL,
                oponente_rank INTEGER,
                superficie TEXT,
                torneo TEXT,
                ronda TEXT,
                
                prob_modelo REAL NOT NULL,
                prob_modelo_calibrada REAL,
                
                cuota REAL NOT NULL,
                bookmaker TEXT,
                ev REAL NOT NULL,
                umbral_ev REAL DEFAULT 0.03,
                
                decision TEXT NOT NULL,
                apuesta_cantidad REAL,
                
                resultado_real INTEGER,
                ganancia REAL,
                fecha_resultado DATETIME,
                
                modelo_usado TEXT,
                version_modelo TEXT,
                notas TEXT
            )
        """
        )

        # Tabla: Configuración
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS configuracion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parametro TEXT UNIQUE NOT NULL,
                valor TEXT NOT NULL,
                fecha_modificacion DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Tabla: Estadísticas diarias
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS estadisticas_diarias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha DATE UNIQUE NOT NULL,
                predicciones_totales INTEGER DEFAULT 0,
                apuestas_realizadas INTEGER DEFAULT 0,
                apuestas_ganadas INTEGER DEFAULT 0,
                apuestas_perdidas INTEGER DEFAULT 0,
                apuestas_pendientes INTEGER DEFAULT 0,
                total_apostado REAL DEFAULT 0,
                total_ganado REAL DEFAULT 0,
                roi REAL,
                ev_promedio REAL,
                win_rate REAL
            )
        """
        )

        # Índices para búsquedas rápidas
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_fecha_partido 
            ON predicciones(fecha_partido)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decision 
            ON predicciones(decision)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_resultado 
            ON predicciones(resultado_real)
        """
        )

        conn.commit()
        logger.info(f"✅ TennisDatabase SQLite schema initialized: {self.db_path}")

    def insertar_prediccion(self, prediccion_dict):
        """
        Inserta una nueva predicción en la base de datos

        Args:
            prediccion_dict: diccionario con todos los campos de la predicción

        Returns:
            id de la predicción insertada
        """
        # Campos obligatorios
        campos_obligatorios = [
            "fecha_partido",
            "jugador_nombre",
            "oponente_nombre",
            "prob_modelo",
            "cuota",
            "ev",
            "decision",
        ]

        for campo in campos_obligatorios:
            if campo not in prediccion_dict:
                raise ValueError(f"Campo obligatorio faltante: {campo}")

        # Añadir fecha de predicción
        prediccion_dict["fecha_prediccion"] = datetime.now().isoformat()

        if self.is_postgres:
            from sqlalchemy import text
            
            # Preparar query para PostgreSQL
            campos = ", ".join(prediccion_dict.keys())
            placeholders = ", ".join([f":{k}" for k in prediccion_dict.keys()])
            
            query = f"""
                INSERT INTO predicciones ({campos})
                VALUES ({placeholders})
                RETURNING id
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), prediccion_dict)
                prediccion_id = result.fetchone()[0]
                conn.commit()
        else:
            conn = self.conectar()
            cursor = conn.cursor()
            
            # Preparar query para SQLite
            campos = ", ".join(prediccion_dict.keys())
            placeholders = ", ".join(["?" for _ in prediccion_dict])

            query = f"""
                INSERT INTO predicciones ({campos})
                VALUES ({placeholders})
            """

            cursor.execute(query, list(prediccion_dict.values()))
            conn.commit()
            prediccion_id = cursor.lastrowid

        logger.info(f"✅ Predicción guardada (ID: {prediccion_id})")
        return prediccion_id

    def actualizar_resultado(self, prediccion_id, resultado, ganancia=None):
        """
        Actualiza el resultado real de una predicción

        Args:
            prediccion_id: ID de la predicción
            resultado: 1 (ganó) o 0 (perdió)
            ganancia: ganancia neta (puede ser negativa)
        """
        self._execute(
            """
            UPDATE predicciones
            SET resultado_real = :resultado,
                ganancia = :ganancia,
                fecha_resultado = :fecha
            WHERE id = :id
            """,
            {
                "resultado": resultado,
                "ganancia": ganancia,
                "fecha": datetime.now().isoformat(),
                "id": prediccion_id,
            },
        )

        logger.info(f"✅ Resultado actualizado (ID: {prediccion_id})")

    def obtener_predicciones(self, filtro=None):
        """
        Obtiene predicciones con filtros opcionales

        Args:
            filtro: dict con filtros, ej: {'decision': 'APOSTAR', 'resultado_real': 1}

        Returns:
            DataFrame con predicciones
        """
        if self.is_postgres:
            query = "SELECT * FROM predicciones"
            
            if filtro:
                condiciones = " AND ".join([f"{k} = :{k}" for k in filtro.keys()])
                query += f" WHERE {condiciones}"
            
            rows = self._fetchall(query, filtro)
            df = pd.DataFrame(rows)
        else:
            query = "SELECT * FROM predicciones"

            if filtro:
                condiciones = " AND ".join([f"{k} = ?" for k in filtro.keys()])
                query += f" WHERE {condiciones}"
                valores = list(filtro.values())
            else:
                valores = []

            df = pd.read_sql_query(query, self.conectar(), params=valores)

        return df

    def calcular_metricas(self):
        """
        Calcula métricas generales
        """
        query = """
            SELECT *
            FROM predicciones
            WHERE decision = 'APOSTAR'
              AND resultado_real IS NOT NULL
        """
        
        if self.is_postgres:
            rows = self._fetchall(query)
            df_apuestas = pd.DataFrame(rows)
        else:
            conn = self.conectar()
            df_apuestas = pd.read_sql_query(query, conn)

        if len(df_apuestas) == 0:
            return {
                "total_apuestas": 0,
                "apuestas_ganadas": 0,
                "apuestas_perdidas": 0,
                "win_rate": 0,
                "total_apostado": 0,
                "total_ganado": 0,
                "ganancia_neta": 0,
                "roi": 0,
                "ev_promedio": 0,
            }

        total_apuestas = len(df_apuestas)
        ganadas = (df_apuestas["resultado_real"] == 1).sum()
        win_rate = ganadas / total_apuestas

        total_apostado = df_apuestas["apuesta_cantidad"].sum() or 0
        ganancia_neta = df_apuestas["ganancia"].sum() or 0
        total_ganado = total_apostado + ganancia_neta  # Retorno total
        roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0

        ev_promedio = df_apuestas["ev"].mean() * 100 if len(df_apuestas) > 0 else 0

        return {
            "total_apuestas": total_apuestas,
            "apuestas_ganadas": int(ganadas),
            "apuestas_perdidas": total_apuestas - int(ganadas),
            "win_rate": win_rate * 100,
            "total_apostado": total_apostado,
            "total_ganado": total_ganado,
            "ganancia_neta": ganancia_neta,
            "roi": roi,
            "ev_promedio": ev_promedio,
        }

    def exportar_a_csv(self, output_path="export_predicciones.csv"):
        """
        Exporta todas las predicciones a CSV
        """
        df = self.obtener_predicciones()
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Predicciones exportadas a: {output_path}")

    def registrar_prediccion(
        self,
        jugador1: str,
        jugador2: str,
        superficie: str,
        probabilidad: float,
        cuota: float,
        decision: str,
        fecha_partido=None,
        **kwargs
    ):
        """
        Método de conveniencia para registrar una predicción
        (compatible con scripts existentes)
        """
        prediccion_dict = {
            "fecha_partido": fecha_partido or datetime.now().date().isoformat(),
            "jugador_nombre": jugador1,
            "oponente_nombre": jugador2,
            "superficie": superficie,
            "prob_modelo": probabilidad,
            "cuota": cuota,
            "ev": (probabilidad * cuota) - 1,
            "decision": decision,
        }
        prediccion_dict.update(kwargs)
        return self.insertar_prediccion(prediccion_dict)


# Ejemplo de uso
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear database
    db = TennisDatabase("apuestas_tracker.db")

    # Ejemplo: Insertar predicción
    prediccion = {
        "fecha_partido": "2024-12-10",
        "jugador_nombre": "Carlos Alcaraz",
        "jugador_rank": 3,
        "oponente_nombre": "Jannik Sinner",
        "oponente_rank": 1,
        "superficie": "Hard",
        "torneo": "ATP Finals",
        "prob_modelo": 0.482,
        "prob_modelo_calibrada": 0.478,
        "cuota": 2.10,
        "bookmaker": "Bet365",
        "ev": 0.0038,
        "decision": "NO_APOSTAR",
        "apuesta_cantidad": None,
        "modelo_usado": "XGBoost Optimizado",
        "version_modelo": "v3.0",
    }

    pred_id = db.insertar_prediccion(prediccion)

    # Calcular métricas
    metricas = db.calcular_metricas()
    print("\n📊 Métricas:")
    for key, value in metricas.items():
        print(f"   {key}: {value}")

    db.desconectar()
