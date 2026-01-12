"""
Sistema de Base de Datos SQLite para Tracking de Predicciones
Fase 4: Tennis ML Predictor
"""

import sqlite3
from datetime import datetime
import pandas as pd


class TennisDatabase:
    """
    Sistema de base de datos para tracking de predicciones y apuestas
    """

    def __init__(self, db_path="apuestas_tracker.db"):
        """
        Inicializa la base de datos

        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.db_path = db_path
        self.conn = None

        # Crear DB si no existe
        self.crear_database()

    def conectar(self):
        """Conectar a la base de datos"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def desconectar(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def crear_database(self):
        """
        Crea todas las tablas necesarias
        """

        conn = self.conectar()
        cursor = conn.cursor()

        # Tabla: Predicciones
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_partido DATE NOT NULL,
                fecha_prediccion DATETIME NOT NULL,
                
                -- Información del partido
                jugador_nombre TEXT NOT NULL,
                jugador_rank INTEGER,
                oponente_nombre TEXT NOT NULL,
                oponente_rank INTEGER,
                superficie TEXT,
                torneo TEXT,
                ronda TEXT,
                
                -- Predicción del modelo
                prob_modelo REAL NOT NULL,
                prob_modelo_calibrada REAL,
                
                -- Información de apuestas
                cuota REAL NOT NULL,
                bookmaker TEXT,
                ev REAL NOT NULL,
                umbral_ev REAL DEFAULT 0.03,
                
                -- Decisión
                decision TEXT NOT NULL,  -- 'APOSTAR' o 'NO_APOSTAR'
                apuesta_cantidad REAL,
                
                -- Resultado real (se actualiza después del partido)
                resultado_real INTEGER,  -- 1=ganó, 0=perdió, NULL=pendiente
                ganancia REAL,
                fecha_resultado DATETIME,
                
                -- Metadata
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
        print("✅ Base de datos creada/actualizada correctamente")
        print(f"📁 Ubicación: {self.db_path}")

    def insertar_prediccion(self, prediccion_dict):
        """
        Inserta una nueva predicción en la base de datos

        Args:
            prediccion_dict: diccionario con todos los campos de la predicción

        Returns:
            id de la predicción insertada
        """

        conn = self.conectar()
        cursor = conn.cursor()

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

        # Preparar query
        campos = ", ".join(prediccion_dict.keys())
        placeholders = ", ".join(["?" for _ in prediccion_dict])

        query = f"""
            INSERT INTO predicciones ({campos})
            VALUES ({placeholders})
        """

        cursor.execute(query, list(prediccion_dict.values()))
        conn.commit()

        prediccion_id = cursor.lastrowid
        print(f"✅ Predicción guardada (ID: {prediccion_id})")

        return prediccion_id

    def actualizar_resultado(self, prediccion_id, resultado, ganancia=None):
        """
        Actualiza el resultado real de una predicción

        Args:
            prediccion_id: ID de la predicción
            resultado: 1 (ganó) o 0 (perdió)
            ganancia: ganancia neta (puede ser negativa)
        """

        conn = self.conectar()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE predicciones
            SET resultado_real = ?,
                ganancia = ?,
                fecha_resultado = ?
            WHERE id = ?
        """,
            (resultado, ganancia, datetime.now().isoformat(), prediccion_id),
        )

        conn.commit()
        print(f"✅ Resultado actualizado (ID: {prediccion_id})")

    def obtener_predicciones(self, filtro=None):
        """
        Obtiene predicciones con filtros opcionales

        Args:
            filtro: dict con filtros, ej: {'decision': 'APOSTAR', 'resultado_real': 1}

        Returns:
            DataFrame con predicciones
        """

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

        conn = self.conectar()

        # Apuestas realizadas
        df_apuestas = pd.read_sql_query(
            """
            SELECT *
            FROM predicciones
            WHERE decision = 'APOSTAR'
              AND resultado_real IS NOT NULL
        """,
            conn,
        )

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

        total_apostado = df_apuestas["apuesta_cantidad"].sum()
        ganancia_neta = df_apuestas["ganancia"].sum()
        total_ganado = total_apostado + ganancia_neta  # Retorno total
        roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0

        ev_promedio = df_apuestas["ev"].mean() * 100

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
        print(f"✅ Predicciones exportadas a: {output_path}")


# Ejemplo de uso
if __name__ == "__main__":
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
