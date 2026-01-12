"""
Calculador de métricas de fatiga
Analiza carga de partidos y descanso de jugadores
"""

import pandas as pd
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FatigaCalculator:
    """
    Calcula features relacionadas con fatiga y carga de partidos

    Por qué son importantes:
    - Djokovic con 2 días de descanso ≠ Djokovic con 10 días
    - Jugador que jugó 4 partidos en 6 días estará más cansado
    - Crítico en torneos largos (Grand Slams, Masters)
    """

    def __init__(self, df_partidos):
        """
        Args:
            df_partidos: DataFrame con partidos
                Columnas requeridas: tourney_date, winner_name, loser_name
                Columnas opcionales: minutes
        """
        self.df = df_partidos.copy()
        self.df["tourney_date"] = pd.to_datetime(self.df["tourney_date"])

    def calcular_fatiga(self, jugador_nombre, fecha_partido):
        """
        Calcula métricas de fatiga

        Features:
        - Días desde último partido
        - Partidos en últimos 7, 14, 30 días
        - Minutos jugados últimos 7, 14 días
        - ¿Viene de torneo largo?
        - ¿Bien descansado?
        - ¿Sin ritmo? (mucho descanso)

        Args:
            jugador_nombre: Nombre del jugador
            fecha_partido: Fecha del partido a predecir

        Returns:
            dict con métricas de fatiga
        """

        # Partidos previos donde jugó (ganó o perdió)
        partidos_ganados = self.df[
            (self.df["winner_name"] == jugador_nombre) & (self.df["tourney_date"] < fecha_partido)
        ][["tourney_date", "minutes"]].copy()

        partidos_perdidos = self.df[
            (self.df["loser_name"] == jugador_nombre) & (self.df["tourney_date"] < fecha_partido)
        ][["tourney_date", "minutes"]].copy()

        # Combinar y ordenar
        partidos_previos = pd.concat([partidos_ganados, partidos_perdidos]).sort_values(
            "tourney_date", ascending=False
        )

        if len(partidos_previos) == 0:
            return self._fatiga_default()

        # Último partido
        ultimo_partido_fecha = partidos_previos.iloc[0]["tourney_date"]
        dias_desde_ultimo = (fecha_partido - ultimo_partido_fecha).days

        # Últimos 7 días
        fecha_7d = fecha_partido - timedelta(days=7)
        partidos_7d = partidos_previos[partidos_previos["tourney_date"] >= fecha_7d]

        # Últimos 14 días
        fecha_14d = fecha_partido - timedelta(days=14)
        partidos_14d = partidos_previos[partidos_previos["tourney_date"] >= fecha_14d]

        # Últimos 30 días
        fecha_30d = fecha_partido - timedelta(days=30)
        partidos_30d = partidos_previos[partidos_previos["tourney_date"] >= fecha_30d]

        # Minutos jugados
        minutos_7d = partidos_7d["minutes"].fillna(120).sum()
        minutos_14d = partidos_14d["minutes"].fillna(120).sum()

        fatiga = {
            # Días de descanso
            "dias_desde_ultimo_partido": dias_desde_ultimo,
            "dias_desde_ultimo_normalizado": min(dias_desde_ultimo / 14, 1.0),  # 0-1
            # Carga reciente (número de partidos)
            "partidos_ultimos_7d": len(partidos_7d),
            "partidos_ultimos_14d": len(partidos_14d),
            "partidos_ultimos_30d": len(partidos_30d),
            # Minutos jugados
            "minutos_ultimos_7d": float(minutos_7d),
            "minutos_ultimos_14d": float(minutos_14d),
            "minutos_por_partido_7d": (
                float(minutos_7d / len(partidos_7d)) if len(partidos_7d) > 0 else 0.0
            ),
            # Indicadores binarios
            "torneo_largo": 1 if len(partidos_7d) >= 4 else 0,  # 4+ partidos en 7 días
            "bien_descansado": 1 if dias_desde_ultimo >= 5 else 0,  # 5+ días descanso
            "sin_ritmo": 1 if dias_desde_ultimo >= 21 else 0,  # 21+ días sin jugar
            "recien_jugado": 1 if dias_desde_ultimo <= 1 else 0,  # Jugó ayer
            # Score de carga (0-1, más alto = más cargado)
            "carga_reciente_score": min((len(partidos_7d) / 5 + len(partidos_14d) / 10) / 2, 1.0),
        }

        return fatiga

    def calcular_ventaja_fatiga(self, jugador1_nombre, jugador2_nombre, fecha_partido):
        """
        Compara fatiga entre dos jugadores

        Args:
            jugador1_nombre: Nombre del jugador 1
            jugador2_nombre: Nombre del jugador 2
            fecha_partido: Fecha del partido

        Returns:
            dict con ventajas relativas
        """

        fatiga_j1 = self.calcular_fatiga(jugador1_nombre, fecha_partido)
        fatiga_j2 = self.calcular_fatiga(jugador2_nombre, fecha_partido)

        ventaja = {
            # Diferencias
            "diff_dias_descanso": fatiga_j1["dias_desde_ultimo_partido"]
            - fatiga_j2["dias_desde_ultimo_partido"],
            "diff_partidos_7d": fatiga_j1["partidos_ultimos_7d"] - fatiga_j2["partidos_ultimos_7d"],
            "diff_partidos_14d": fatiga_j1["partidos_ultimos_14d"]
            - fatiga_j2["partidos_ultimos_14d"],
            "diff_minutos_7d": fatiga_j1["minutos_ultimos_7d"] - fatiga_j2["minutos_ultimos_7d"],
            "diff_carga_score": fatiga_j1["carga_reciente_score"]
            - fatiga_j2["carga_reciente_score"],
            # Indicadores
            "j1_mas_fresco": (
                1
                if fatiga_j1["dias_desde_ultimo_partido"] > fatiga_j2["dias_desde_ultimo_partido"]
                else 0
            ),
            "j1_mas_activo": (
                1 if fatiga_j1["partidos_ultimos_14d"] > fatiga_j2["partidos_ultimos_14d"] else 0
            ),
            # Situaciones especiales
            "ambos_cansados": (
                1 if (fatiga_j1["torneo_largo"] == 1 and fatiga_j2["torneo_largo"] == 1) else 0
            ),
            "j1_torneo_largo_j2_no": (
                1 if (fatiga_j1["torneo_largo"] == 1 and fatiga_j2["torneo_largo"] == 0) else 0
            ),
        }

        return ventaja

    def _fatiga_default(self):
        """Valores por defecto (jugador promedio)"""
        return {
            "dias_desde_ultimo_partido": 7,
            "dias_desde_ultimo_normalizado": 0.5,
            "partidos_ultimos_7d": 1,
            "partidos_ultimos_14d": 2,
            "partidos_ultimos_30d": 4,
            "minutos_ultimos_7d": 120.0,
            "minutos_ultimos_14d": 240.0,
            "minutos_por_partido_7d": 120.0,
            "torneo_largo": 0,
            "bien_descansado": 0,
            "sin_ritmo": 0,
            "recien_jugado": 0,
            "carga_reciente_score": 0.3,
        }


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))

    logger.info("Cargando datos...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    calc = FatigaCalculator(df)

    # Buscar jugador de ejemplo
    jugadores_unicos = pd.concat([df["winner_name"], df["loser_name"]]).unique()
    jugador_ejemplo = None
    for nombre in ["Novak Djokovic", "Carlos Alcaraz", "Rafael Nadal"]:
        if nombre in jugadores_unicos:
            jugador_ejemplo = nombre
            break

    if jugador_ejemplo is None:
        jugador_ejemplo = jugadores_unicos[0]

    fecha_ejemplo = df["tourney_date"].max() - pd.Timedelta(days=30)

    logger.info("=" * 60)
    logger.info(f"😴 MÉTRICAS DE FATIGA - {jugador_ejemplo}")
    logger.info("=" * 60)

    fatiga = calc.calcular_fatiga(jugador_nombre=jugador_ejemplo, fecha_partido=fecha_ejemplo)

    for key, value in fatiga.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")

    logger.info("\n✅ Proceso completado!")
