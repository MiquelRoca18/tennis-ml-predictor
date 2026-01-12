"""
Calculador de Head-to-Head mejorado
Analiza enfrentamientos previos entre jugadores
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeadToHeadCalculator:
    """
    Calcula features detalladas de enfrentamientos previos
    """

    def __init__(self, df_partidos):
        """
        Args:
            df_partidos: DataFrame con partidos
                Columnas requeridas: tourney_date, winner_name, loser_name, surface
        """
        self.df = df_partidos.copy()
        self.df["tourney_date"] = pd.to_datetime(self.df["tourney_date"])

    def calcular_h2h(self, jugador1_nombre, jugador2_nombre, fecha_partido, superficie=None):
        """
        Calcula H2H completo entre dos jugadores

        Args:
            jugador1_nombre: Nombre del jugador 1
            jugador2_nombre: Nombre del jugador 2
            fecha_partido: Fecha del partido a predecir
            superficie: 'Hard', 'Clay', 'Grass' (None = todas)

        Returns:
            dict con features H2H
        """

        # Filtrar enfrentamientos previos
        h2h_matches = self.df[
            (
                (
                    (self.df["winner_name"] == jugador1_nombre)
                    & (self.df["loser_name"] == jugador2_nombre)
                )
                | (
                    (self.df["winner_name"] == jugador2_nombre)
                    & (self.df["loser_name"] == jugador1_nombre)
                )
            )
            & (self.df["tourney_date"] < fecha_partido)
        ].copy()

        if len(h2h_matches) == 0:
            return self._h2h_default()

        # H2H general
        victorias_j1 = len(h2h_matches[h2h_matches["winner_name"] == jugador1_nombre])
        total_partidos = len(h2h_matches)
        h2h_win_rate = victorias_j1 / total_partidos

        # H2H en esta superficie
        h2h_superficie_rate = 0.5
        if superficie:
            h2h_superficie = h2h_matches[h2h_matches["surface"] == superficie]
            if len(h2h_superficie) > 0:
                victorias_superficie = len(
                    h2h_superficie[h2h_superficie["winner_name"] == jugador1_nombre]
                )
                h2h_superficie_rate = victorias_superficie / len(h2h_superficie)

        # H2H reciente (últimos 2 años)
        fecha_limite = fecha_partido - pd.Timedelta(days=730)
        h2h_reciente = h2h_matches[h2h_matches["tourney_date"] >= fecha_limite]

        if len(h2h_reciente) > 0:
            victorias_recientes = len(h2h_reciente[h2h_reciente["winner_name"] == jugador1_nombre])
            h2h_reciente_rate = victorias_recientes / len(h2h_reciente)
        else:
            h2h_reciente_rate = h2h_win_rate

        return {
            "h2h_total_partidos": total_partidos,
            "h2h_victorias": victorias_j1,
            "h2h_win_rate": h2h_win_rate,
            "h2h_superficie_rate": h2h_superficie_rate,
            "h2h_reciente_rate": h2h_reciente_rate,
        }

    def _h2h_default(self):
        """Valores por defecto sin enfrentamientos previos"""
        return {
            "h2h_total_partidos": 0,
            "h2h_victorias": 0,
            "h2h_win_rate": 0.5,
            "h2h_superficie_rate": 0.5,
            "h2h_reciente_rate": 0.5,
        }


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))

    logger.info("Cargando datos...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    calc = HeadToHeadCalculator(df)

    # Buscar dos jugadores que se hayan enfrentado
    jugadores_unicos = pd.concat([df["winner_name"], df["loser_name"]]).unique()

    # Intentar encontrar un par conocido
    pares_conocidos = [
        ("Carlos Alcaraz", "Jannik Sinner"),
        ("Novak Djokovic", "Rafael Nadal"),
        ("Roger Federer", "Rafael Nadal"),
        ("Novak Djokovic", "Roger Federer"),
    ]

    jugador1, jugador2 = None, None
    for j1, j2 in pares_conocidos:
        if j1 in jugadores_unicos and j2 in jugadores_unicos:
            jugador1, jugador2 = j1, j2
            break

    if jugador1 is None:
        # Tomar dos jugadores cualesquiera que se hayan enfrentado
        for idx, row in df.head(100).iterrows():
            j1, j2 = row["winner_name"], row["loser_name"]
            # Verificar que se hayan enfrentado más de una vez
            enfrentamientos = df[
                ((df["winner_name"] == j1) & (df["loser_name"] == j2))
                | ((df["winner_name"] == j2) & (df["loser_name"] == j1))
            ]
            if len(enfrentamientos) > 1:
                jugador1, jugador2 = j1, j2
                break

    if jugador1 and jugador2:
        fecha_ejemplo = df["tourney_date"].max() - pd.Timedelta(days=30)

        logger.info("=" * 60)
        logger.info(f"🎾 HEAD-TO-HEAD: {jugador1} vs {jugador2}")
        logger.info("=" * 60)

        h2h = calc.calcular_h2h(
            jugador1_nombre=jugador1,
            jugador2_nombre=jugador2,
            fecha_partido=fecha_ejemplo,
            superficie="Hard",
        )

        for key, value in h2h.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")
    else:
        logger.info("No se encontraron jugadores con enfrentamientos previos")

    logger.info("\n✅ Proceso completado!")
