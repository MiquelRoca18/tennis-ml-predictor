"""
Match Database Manager - Tennis ML Predictor v2.0
==================================================

Gestión de base de datos para partidos con predicciones versionadas,
tracking de apuestas y estadísticas.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MatchDatabase:
    """
    Gestor de base de datos para partidos de tenis con predicciones y apuestas
    """
    
    def __init__(self, db_path: str = "matches_v2.db"):
        """
        Inicializa la conexión a la base de datos
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()
        self._initialize_schema()
        logger.info(f"✅ MatchDatabase inicializada: {self.db_path}")
    
    def _connect(self):
        """Establece conexión con la base de datos"""
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        
    def _initialize_schema(self):
        """Crea las tablas si no existen"""
        # Verificar si las tablas ya existen
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='matches'
        """)
        
        if cursor.fetchone():
            logger.info("✅ Esquema ya existe, saltando inicialización")
            return
        
        # Si no existen, crear desde schema
        schema_path = Path(__file__).parent / "schema_v2.sql"
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
                self.conn.executescript(schema_sql)
                self.conn.commit()
                logger.info("✅ Esquema de base de datos inicializado")
        else:
            logger.warning(f"⚠️  No se encontró schema_v2.sql en {schema_path}")
    
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
        jugador2_ranking: Optional[int] = None
    ) -> int:
        """
        Crea un nuevo partido
        
        Returns:
            ID del partido creado
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO matches (
                fecha_partido, hora_inicio, torneo, ronda, superficie,
                jugador1_nombre, jugador1_ranking,
                jugador2_nombre, jugador2_ranking,
                estado
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pendiente')
        """, (
            fecha_partido, hora_inicio, torneo, ronda, superficie,
            jugador1_nombre, jugador1_ranking,
            jugador2_nombre, jugador2_ranking
        ))
        
        self.conn.commit()
        match_id = cursor.lastrowid
        
        logger.info(f"✅ Partido creado: {jugador1_nombre} vs {jugador2_nombre} (ID: {match_id})")
        return match_id
    
    def get_match(self, match_id: int) -> Optional[Dict]:
        """Obtiene un partido por ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_matches_by_date(self, fecha: date) -> List[Dict]:
        """
        Obtiene todos los partidos de una fecha específica
        
        Returns:
            Lista de partidos con sus predicciones y resultados
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM matches_with_latest_prediction
            WHERE fecha_partido = ?
            ORDER BY hora_inicio ASC, id ASC
        """, (fecha,))
        
        matches = [dict(row) for row in cursor.fetchall()]
        logger.info(f"📊 Encontrados {len(matches)} partidos para {fecha}")
        return matches
    
    def update_match_result(
        self,
        match_id: int,
        ganador: str,
        marcador: Optional[str] = None
    ) -> bool:
        """
        Actualiza el resultado de un partido
        
        Returns:
            True si se actualizó correctamente
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE matches
            SET resultado_ganador = ?,
                resultado_marcador = ?,
                estado = 'completado'
            WHERE id = ?
        """, (ganador, marcador, match_id))
        
        self.conn.commit()
        
        if cursor.rowcount > 0:
            logger.info(f"✅ Resultado actualizado para partido {match_id}: Ganador {ganador}")
            return True
        
        logger.warning(f"⚠️  No se pudo actualizar resultado del partido {match_id}")
        return False
    
    # ============================================================
    # MÉTODOS DE PREDICCIONES
    # ============================================================
    
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
        kelly_stake_jugador2: Optional[float] = None
    ) -> int:
        """
        Añade una nueva versión de predicción para un partido
        
        Returns:
            ID de la predicción creada
        """
        cursor = self.conn.cursor()
        
        # Obtener la última versión
        cursor.execute("""
            SELECT COALESCE(MAX(version), 0) as max_version
            FROM predictions
            WHERE match_id = ?
        """, (match_id,))
        
        max_version = cursor.fetchone()['max_version']
        new_version = max_version + 1
        
        # Insertar nueva predicción
        cursor.execute("""
            INSERT INTO predictions (
                match_id, version,
                jugador1_cuota, jugador2_cuota,
                jugador1_probabilidad, jugador2_probabilidad,
                jugador1_ev, jugador2_ev,
                jugador1_edge, jugador2_edge,
                recomendacion, mejor_opcion, confianza,
                kelly_stake_jugador1, kelly_stake_jugador2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, new_version,
            jugador1_cuota, jugador2_cuota,
            jugador1_probabilidad, jugador2_probabilidad,
            jugador1_ev, jugador2_ev,
            jugador1_edge, jugador2_edge,
            recomendacion, mejor_opcion, confianza,
            kelly_stake_jugador1, kelly_stake_jugador2
        ))
        
        self.conn.commit()
        prediction_id = cursor.lastrowid
        
        logger.info(f"✅ Predicción v{new_version} creada para partido {match_id}: {recomendacion}")
        return prediction_id
    
    def get_latest_prediction(self, match_id: int) -> Optional[Dict]:
        """Obtiene la última predicción de un partido"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE match_id = ?
            ORDER BY version DESC
            LIMIT 1
        """, (match_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_prediction_history(self, match_id: int) -> List[Dict]:
        """Obtiene todas las versiones de predicción de un partido"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE match_id = ?
            ORDER BY version ASC
        """, (match_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ============================================================
    # MÉTODOS DE APUESTAS (BETS)
    # ============================================================
    
    def register_bet(
        self,
        match_id: int,
        prediction_id: int,
        jugador_apostado: str,
        cuota_apostada: float,
        stake: float
    ) -> int:
        """
        Registra una nueva apuesta
        
        Returns:
            ID de la apuesta creada
        """
        cursor = self.conn.cursor()
        
        # Verificar si ya existe una apuesta activa para este partido
        cursor.execute("""
            SELECT id FROM bets
            WHERE match_id = ? AND estado = 'activa'
        """, (match_id,))
        
        existing_bet = cursor.fetchone()
        if existing_bet:
            logger.warning(f"⚠️  Ya existe una apuesta activa para el partido {match_id}")
            return existing_bet['id']
        
        # Crear nueva apuesta
        cursor.execute("""
            INSERT INTO bets (
                match_id, prediction_id,
                jugador_apostado, cuota_apostada, stake,
                estado
            ) VALUES (?, ?, ?, ?, ?, 'activa')
        """, (match_id, prediction_id, jugador_apostado, cuota_apostada, stake))
        
        self.conn.commit()
        bet_id = cursor.lastrowid
        
        logger.info(f"✅ Apuesta registrada: {jugador_apostado} @ {cuota_apostada} (Stake: {stake}€)")
        return bet_id
    
    def cancel_bet(self, match_id: int) -> bool:
        """
        Cancela la apuesta activa de un partido
        
        Returns:
            True si se canceló correctamente
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE bets
            SET estado = 'cancelada',
                resultado = 'cancelada'
            WHERE match_id = ? AND estado = 'activa'
        """, (match_id,))
        
        self.conn.commit()
        
        if cursor.rowcount > 0:
            logger.info(f"✅ Apuesta cancelada para partido {match_id}")
            return True
        
        return False
    
    def update_bet_result(self, match_id: int, ganador: str) -> bool:
        """
        Actualiza el resultado de una apuesta cuando el partido termina
        
        Returns:
            True si se actualizó correctamente
        """
        cursor = self.conn.cursor()
        
        # Obtener la apuesta activa
        cursor.execute("""
            SELECT * FROM bets
            WHERE match_id = ? AND estado = 'activa'
        """, (match_id,))
        
        bet = cursor.fetchone()
        if not bet:
            logger.warning(f"⚠️  No hay apuesta activa para el partido {match_id}")
            return False
        
        # Calcular resultado
        bet_dict = dict(bet)
        jugador_apostado = bet_dict['jugador_apostado']
        cuota = bet_dict['cuota_apostada']
        stake = bet_dict['stake']
        
        if jugador_apostado == ganador:
            # Apuesta ganada
            resultado = 'ganada'
            ganancia = stake * (cuota - 1)  # Ganancia neta
            roi = (cuota - 1)
        else:
            # Apuesta perdida
            resultado = 'perdida'
            ganancia = -stake
            roi = -1.0
        
        # Actualizar apuesta
        cursor.execute("""
            UPDATE bets
            SET resultado = ?,
                ganancia = ?,
                roi = ?,
                estado = 'completada'
            WHERE id = ?
        """, (resultado, ganancia, roi, bet_dict['id']))
        
        self.conn.commit()
        
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
        cursor = self.conn.cursor()
        
        # Filtro de fecha
        if days:
            fecha_inicio = date.today() - timedelta(days=days)
            date_filter = f"AND DATE(b.timestamp_apuesta) >= '{fecha_inicio}'"
        else:
            fecha_inicio = None
            date_filter = ""
        
        # Estadísticas de apuestas
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN resultado = 'ganada' THEN 1 ELSE 0 END) as ganadas,
                SUM(CASE WHEN resultado = 'perdida' THEN 1 ELSE 0 END) as perdidas,
                SUM(stake) as stake_total,
                SUM(ganancia) as ganancia_neta
            FROM bets b
            WHERE estado = 'completada'
            {date_filter}
        """)
        
        stats = dict(cursor.fetchone())
        
        # Calcular métricas derivadas
        total = stats['total'] or 0
        ganadas = stats['ganadas'] or 0
        stake_total = stats['stake_total'] or 0
        ganancia_neta = stats['ganancia_neta'] or 0
        
        win_rate = ganadas / total if total > 0 else 0.0
        roi = ganancia_neta / stake_total if stake_total > 0 else 0.0
        ganancia_bruta = ganancia_neta + stake_total if ganancia_neta >= 0 else 0
        
        return {
            'periodo': f'Últimos {days} días' if days else 'Todos los tiempos',
            'fecha_inicio': fecha_inicio,
            'fecha_fin': date.today(),
            'apuestas': {
                'total': total,
                'ganadas': ganadas,
                'perdidas': stats['perdidas'] or 0,
                'win_rate': round(win_rate, 3)
            },
            'financiero': {
                'stake_total': round(stake_total, 2),
                'ganancia_bruta': round(ganancia_bruta, 2),
                'ganancia_neta': round(ganancia_neta, 2),
                'roi': round(roi, 3)
            },
            'modelo': {
                'accuracy': None,  # TODO: Calcular desde predictions
                'brier_score': None,
                'ev_promedio': None
            }
        }
    
    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """
        Obtiene estadísticas diarias para gráficos
        
        Returns:
            Lista de estadísticas por día
        """
        cursor = self.conn.cursor()
        
        fecha_inicio = date.today() - timedelta(days=days)
        
        cursor.execute("""
            SELECT * FROM daily_stats
            WHERE fecha >= ?
            ORDER BY fecha DESC
        """, (fecha_inicio,))
        
        return [dict(row) for row in cursor.fetchall()]
    
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
        ronda="Semifinal"
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
        kelly_stake_jugador1=12.5
    )
    
    # Registrar apuesta
    bet_id = db.register_bet(
        match_id=match_id,
        prediction_id=prediction_id,
        jugador_apostado="Djokovic",
        cuota_apostada=1.85,
        stake=12.5
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
