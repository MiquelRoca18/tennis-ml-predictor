"""
Features de Edad y Experiencia del Jugador
Captura patrones relacionados con edad, madurez y experiencia en el tour
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgeExperienceCalculator:
    """
    Calcula features relacionadas con edad y experiencia del jugador
    """

    # Rangos de edad óptima para tenis profesional
    PRIME_AGE_MIN = 23
    PRIME_AGE_MAX = 29
    YOUNG_AGE_MAX = 22
    VETERAN_AGE_MIN = 32

    def __init__(self, df_partidos):
        """
        Args:
            df_partidos: DataFrame con partidos históricos
        """
        self.df = df_partidos.copy()

        # Detectar nombre de columna de fecha
        if "fecha" in self.df.columns:
            fecha_col = "fecha"
        elif "tourney_date" in self.df.columns:
            fecha_col = "tourney_date"
            self.df["fecha"] = pd.to_datetime(self.df[fecha_col], format="%Y%m%d")
        else:
            raise ValueError("No se encontró columna de fecha (fecha o tourney_date)")

        if fecha_col == "fecha":
            self.df["fecha"] = pd.to_datetime(self.df["fecha"])

        # Calcular estadísticas de carrera por jugador
        self._calcular_estadisticas_carrera()

        logger.info("✅ AgeExperienceCalculator inicializado")

    def _calcular_estadisticas_carrera(self):
        """Calcula estadísticas de carrera para cada jugador"""
        # Contar partidos por jugador hasta cada fecha
        all_matches = []

        for _, row in self.df.iterrows():
            # Detectar nombres de columnas
            winner_col = "winner_name" if "winner_name" in row else "ganador_nombre"
            loser_col = "loser_name" if "loser_name" in row else "perdedor_nombre"

            all_matches.append({"jugador": row[winner_col], "fecha": row["fecha"], "resultado": 1})
            all_matches.append({"jugador": row[loser_col], "fecha": row["fecha"], "resultado": 0})

        matches_df = pd.DataFrame(all_matches)
        matches_df = matches_df.sort_values("fecha")

        # Calcular partidos acumulados
        self.career_stats = {}
        for jugador in matches_df["jugador"].unique():
            jugador_matches = matches_df[matches_df["jugador"] == jugador].copy()
            jugador_matches["partidos_career"] = range(1, len(jugador_matches) + 1)
            jugador_matches["victorias_career"] = jugador_matches["resultado"].cumsum()
            jugador_matches["win_rate_career"] = (
                jugador_matches["victorias_career"] / jugador_matches["partidos_career"]
            )

            # Primera aparición (debut)
            primera_fecha = jugador_matches["fecha"].min()
            jugador_matches["anos_tour"] = (
                jugador_matches["fecha"] - primera_fecha
            ).dt.days / 365.25

            self.career_stats[jugador] = jugador_matches.set_index("fecha")

        logger.info(f"   Estadísticas calculadas para {len(self.career_stats)} jugadores")

    def calcular_features_edad(self, partido_row):
        """
        Calcula features de edad para un partido

        Args:
            partido_row: Fila del partido con información de edad

        Returns:
            dict con features de edad
        """
        features = {}

        # Edades (ya vienen en los datos TML)
        # Detectar nombres de columnas
        winner_age_col = "winner_age" if "winner_age" in partido_row else "ganador_edad"
        loser_age_col = "loser_age" if "loser_age" in partido_row else "perdedor_edad"

        j1_edad = partido_row.get(winner_age_col, np.nan)
        j2_edad = partido_row.get(loser_age_col, np.nan)

        if pd.notna(j1_edad) and pd.notna(j2_edad):
            # Edades básicas
            features["j1_edad"] = j1_edad
            features["j2_edad"] = j2_edad
            features["diff_edad"] = j1_edad - j2_edad
            features["edad_promedio"] = (j1_edad + j2_edad) / 2

            # Categorías de edad
            features["j1_edad_prime"] = (
                1 if self.PRIME_AGE_MIN <= j1_edad <= self.PRIME_AGE_MAX else 0
            )
            features["j2_edad_prime"] = (
                1 if self.PRIME_AGE_MIN <= j2_edad <= self.PRIME_AGE_MAX else 0
            )

            features["j1_joven"] = 1 if j1_edad <= self.YOUNG_AGE_MAX else 0
            features["j2_joven"] = 1 if j2_edad <= self.YOUNG_AGE_MAX else 0

            features["j1_veterano"] = 1 if j1_edad >= self.VETERAN_AGE_MIN else 0
            features["j2_veterano"] = 1 if j2_edad >= self.VETERAN_AGE_MIN else 0

            # Ventaja de edad
            features["j1_mas_joven"] = 1 if j1_edad < j2_edad else 0
            features["ambos_prime"] = (
                1 if (features["j1_edad_prime"] and features["j2_edad_prime"]) else 0
            )
            features["joven_vs_veterano"] = (
                1 if (features["j1_joven"] and features["j2_veterano"]) else 0
            )
            features["veterano_vs_joven"] = (
                1 if (features["j1_veterano"] and features["j2_joven"]) else 0
            )

        else:
            # Si no hay datos de edad, usar valores por defecto
            for key in [
                "j1_edad",
                "j2_edad",
                "diff_edad",
                "edad_promedio",
                "j1_edad_prime",
                "j2_edad_prime",
                "j1_joven",
                "j2_joven",
                "j1_veterano",
                "j2_veterano",
                "j1_mas_joven",
                "ambos_prime",
                "joven_vs_veterano",
                "veterano_vs_joven",
            ]:
                features[key] = 0 if "edad" not in key else 27  # Edad promedio por defecto

        return features

    def calcular_features_experiencia(self, jugador_nombre, oponente_nombre, fecha_partido):
        """
        Calcula features de experiencia en carrera

        Args:
            jugador_nombre: Nombre del jugador
            oponente_nombre: Nombre del oponente
            fecha_partido: Fecha del partido

        Returns:
            dict con features de experiencia
        """
        features = {}

        # Obtener estadísticas del jugador
        if jugador_nombre in self.career_stats:
            jugador_stats = self.career_stats[jugador_nombre]
            # Buscar stats hasta la fecha del partido
            stats_hasta_fecha = jugador_stats[jugador_stats.index <= fecha_partido]

            if len(stats_hasta_fecha) > 0:
                ultima_stat = stats_hasta_fecha.iloc[-1]
                features["j1_partidos_career"] = ultima_stat["partidos_career"]
                features["j1_victorias_career"] = ultima_stat["victorias_career"]
                features["j1_win_rate_career"] = ultima_stat["win_rate_career"]
                features["j1_anos_tour"] = ultima_stat["anos_tour"]
            else:
                # Primer partido del jugador
                features["j1_partidos_career"] = 0
                features["j1_victorias_career"] = 0
                features["j1_win_rate_career"] = 0.5
                features["j1_anos_tour"] = 0
        else:
            # Jugador no encontrado
            features["j1_partidos_career"] = 0
            features["j1_victorias_career"] = 0
            features["j1_win_rate_career"] = 0.5
            features["j1_anos_tour"] = 0

        # Obtener estadísticas del oponente
        if oponente_nombre in self.career_stats:
            oponente_stats = self.career_stats[oponente_nombre]
            stats_hasta_fecha = oponente_stats[oponente_stats.index <= fecha_partido]

            if len(stats_hasta_fecha) > 0:
                ultima_stat = stats_hasta_fecha.iloc[-1]
                features["j2_partidos_career"] = ultima_stat["partidos_career"]
                features["j2_victorias_career"] = ultima_stat["victorias_career"]
                features["j2_win_rate_career"] = ultima_stat["win_rate_career"]
                features["j2_anos_tour"] = ultima_stat["anos_tour"]
            else:
                features["j2_partidos_career"] = 0
                features["j2_victorias_career"] = 0
                features["j2_win_rate_career"] = 0.5
                features["j2_anos_tour"] = 0
        else:
            features["j2_partidos_career"] = 0
            features["j2_victorias_career"] = 0
            features["j2_win_rate_career"] = 0.5
            features["j2_anos_tour"] = 0

        # Features de diferencia
        features["diff_partidos_career"] = (
            features["j1_partidos_career"] - features["j2_partidos_career"]
        )
        features["diff_anos_tour"] = features["j1_anos_tour"] - features["j2_anos_tour"]
        features["diff_win_rate_career"] = (
            features["j1_win_rate_career"] - features["j2_win_rate_career"]
        )

        # Features categóricas
        features["j1_mas_experimentado"] = (
            1 if features["j1_partidos_career"] > features["j2_partidos_career"] else 0
        )
        features["j1_novato"] = 1 if features["j1_partidos_career"] < 50 else 0
        features["j2_novato"] = 1 if features["j2_partidos_career"] < 50 else 0
        features["novato_vs_veterano"] = (
            1 if (features["j1_novato"] and features["j2_partidos_career"] > 500) else 0
        )

        return features

    def calcular_features_interaccion(self, features_edad, features_exp, partido_row):
        """
        Calcula features de interacción entre edad, experiencia y otras variables

        Args:
            features_edad: Features de edad
            features_exp: Features de experiencia
            partido_row: Fila del partido

        Returns:
            dict con features de interacción
        """
        features = {}

        # Edad × Ranking
        winner_rank_col = "winner_rank" if "winner_rank" in partido_row else "ganador_rank"
        loser_rank_col = "loser_rank" if "loser_rank" in partido_row else "perdedor_rank"

        if winner_rank_col in partido_row and pd.notna(partido_row[winner_rank_col]):
            features["j1_edad_x_rank"] = features_edad.get("j1_edad", 27) * np.log1p(
                partido_row[winner_rank_col]
            )
        else:
            features["j1_edad_x_rank"] = 0

        if loser_rank_col in partido_row and pd.notna(partido_row[loser_rank_col]):
            features["j2_edad_x_rank"] = features_edad.get("j2_edad", 27) * np.log1p(
                partido_row[loser_rank_col]
            )
        else:
            features["j2_edad_x_rank"] = 0

        # Experiencia × Edad (ratio)
        if features_edad.get("j1_edad", 0) > 0:
            features["j1_exp_por_edad"] = features_exp.get(
                "j1_partidos_career", 0
            ) / features_edad.get("j1_edad", 27)
        else:
            features["j1_exp_por_edad"] = 0

        if features_edad.get("j2_edad", 0) > 0:
            features["j2_exp_por_edad"] = features_exp.get(
                "j2_partidos_career", 0
            ) / features_edad.get("j2_edad", 27)
        else:
            features["j2_exp_por_edad"] = 0

        # Años en tour × Win rate career
        features["j1_anos_x_winrate"] = features_exp.get("j1_anos_tour", 0) * features_exp.get(
            "j1_win_rate_career", 0.5
        )
        features["j2_anos_x_winrate"] = features_exp.get("j2_anos_tour", 0) * features_exp.get(
            "j2_win_rate_career", 0.5
        )

        return features


def crear_features_edad_experiencia_partido(age_exp_calc, partido_row):
    """
    Función helper para crear todas las features de edad/experiencia de un partido

    Args:
        age_exp_calc: Instancia de AgeExperienceCalculator
        partido_row: Fila del partido

    Returns:
        dict con todas las features
    """
    features = {}

    # Features de edad
    edad_features = age_exp_calc.calcular_features_edad(partido_row)
    features.update(edad_features)

    # Features de experiencia
    # Detectar nombres de columnas
    winner_name_col = "winner_name" if "winner_name" in partido_row else "ganador_nombre"
    loser_name_col = "loser_name" if "loser_name" in partido_row else "perdedor_nombre"

    exp_features = age_exp_calc.calcular_features_experiencia(
        partido_row.get(winner_name_col, ""),
        partido_row.get(loser_name_col, ""),
        partido_row.get("fecha"),
    )
    features.update(exp_features)

    # Features de interacción
    inter_features = age_exp_calc.calcular_features_interaccion(
        edad_features, exp_features, partido_row
    )
    features.update(inter_features)

    return features
