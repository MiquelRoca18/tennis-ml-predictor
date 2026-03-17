"""
Sistema ELO para tenis
Implementación basada en investigaciones de FiveThirtyEight y adaptado para tenis ATP
"""

import re
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TennisELO:
    """
    Sistema ELO para tenis

    El ELO es superior al ranking ATP porque:
    - Se actualiza después de CADA partido
    - Refleja forma ACTUAL del jugador
    - Se ajusta por calidad del oponente
    - Puede tener ELO diferente por superficie
    """

    def __init__(self, k_factor=32, base_rating=1500):
        """
        Args:
            k_factor: Factor de ajuste (32 es estándar para tenis)
            base_rating: Rating inicial para nuevos jugadores
        """
        self.k_factor = k_factor
        self.base_rating = base_rating

        # Diccionarios para almacenar ratings
        self.ratings = {}  # {jugador: rating}
        self.ratings_by_surface = {"Hard": {}, "Clay": {}, "Grass": {}}

    def get_rating(self, player, surface=None):
        """
        Obtiene rating de un jugador

        Args:
            player: Nombre del jugador
            surface: Si se especifica, devuelve ELO de esa superficie

        Returns:
            Rating ELO (base_rating si es nuevo jugador)
        """
        if surface and surface in self.ratings_by_surface:
            return self.ratings_by_surface[surface].get(player, self.base_rating)
        else:
            return self.ratings.get(player, self.base_rating)

    def expected_score(self, rating_a, rating_b):
        """
        Calcula probabilidad esperada de que A gane vs B

        Formula: E_a = 1 / (1 + 10^((rating_b - rating_a)/400))

        Args:
            rating_a: ELO del jugador A
            rating_b: ELO del jugador B

        Returns:
            Probabilidad de que A gane (0.0 a 1.0)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner, loser, surface=None, margin=None):
        """
        Actualiza ratings después de un partido.

        El ELO global y el ELO por superficie son memorias independientes:
        - El ELO global siempre se actualiza usando los ratings globales.
        - El ELO de superficie se actualiza usando los ratings de esa superficie.
        Nunca se mezclan los dos para el cálculo.

        Args:
            winner: Nombre del ganador
            loser: Nombre del perdedor
            surface: Superficie del partido (opcional)
            margin: Diferencia de sets (sets_ganador - sets_perdedor, opcional)

        Returns:
            dict con ratings nuevos y cambios
        """
        # Ajuste por margen (opcional)
        k = self.k_factor
        if margin and margin > 0:
            # Victorias amplias importan más (pero no demasiado)
            k = k * min(1 + margin / 10, 1.5)

        # --- ELO GLOBAL (siempre se actualiza con ratings globales) ---
        rw_g = self.ratings.get(winner, self.base_rating)
        rl_g = self.ratings.get(loser, self.base_rating)
        exp_w_g = self.expected_score(rw_g, rl_g)

        new_rw_g = rw_g + k * (1 - exp_w_g)
        new_rl_g = rl_g + k * (0 - (1 - exp_w_g))

        self.ratings[winner] = new_rw_g
        self.ratings[loser] = new_rl_g

        # --- ELO POR SUPERFICIE (cálculo independiente del global) ---
        if surface and surface in self.ratings_by_surface:
            rw_s = self.ratings_by_surface[surface].get(winner, self.base_rating)
            rl_s = self.ratings_by_surface[surface].get(loser, self.base_rating)
            exp_w_s = self.expected_score(rw_s, rl_s)

            new_rw_s = rw_s + k * (1 - exp_w_s)
            new_rl_s = rl_s + k * (0 - (1 - exp_w_s))

            self.ratings_by_surface[surface][winner] = new_rw_s
            self.ratings_by_surface[surface][loser] = new_rl_s
        else:
            # Si no hay superficie, los valores de superficie son los globales
            new_rw_s, new_rl_s = new_rw_g, new_rl_g
            rw_s, rl_s = rw_g, rl_g

        return {
            "winner": winner,
            "loser": loser,
            "winner_rating_old": rw_g,
            "winner_rating_new": new_rw_g,
            "winner_change": new_rw_g - rw_g,
            "loser_rating_old": rl_g,
            "loser_rating_new": new_rl_g,
            "loser_change": new_rl_g - rl_g,
            "expected_winner_prob": exp_w_g,
        }

    def calculate_historical_elos(self, df_matches):
        """
        Calcula ELOs históricos para todo el dataset

        IMPORTANTE: El DataFrame debe estar ordenado por fecha

        Args:
            df_matches: DataFrame con partidos ordenados por fecha
                Columnas requeridas: tourney_date, winner_name, loser_name, surface
                Columnas opcionales: score (para calcular margen)

        Returns:
            DataFrame con columnas adicionales de ELO
        """

        logger.info("=" * 60)
        logger.info("📊 CALCULANDO ELOs HISTÓRICOS")
        logger.info("=" * 60)

        # Asegurar que está ordenado por fecha
        df = df_matches.copy()
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
        df = df.sort_values("tourney_date").reset_index(drop=True)

        # Listas para almacenar ELOs en cada partido
        winner_elos = []
        loser_elos = []
        winner_elos_surface = []
        loser_elos_surface = []
        elo_diffs = []
        elo_diffs_surface = []
        expected_probs = []

        total = len(df)
        logger.info(f"\n🔄 Procesando {total:,} partidos...")

        for idx, row in df.iterrows():
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"   Progreso: {idx:,}/{total:,} ({idx/total*100:.1f}%)")

            winner = row["winner_name"]
            loser = row["loser_name"]
            surface = row.get("surface", "Hard")

            # Normalizar superficie
            if surface not in ["Hard", "Clay", "Grass"]:
                surface = "Hard"

            # Obtener ELOs ANTES del partido
            elo_winner = self.get_rating(winner)
            elo_loser = self.get_rating(loser)
            elo_winner_surf = self.get_rating(winner, surface)
            elo_loser_surf = self.get_rating(loser, surface)

            # Guardar
            winner_elos.append(elo_winner)
            loser_elos.append(elo_loser)
            winner_elos_surface.append(elo_winner_surf)
            loser_elos_surface.append(elo_loser_surf)
            elo_diffs.append(elo_winner - elo_loser)
            elo_diffs_surface.append(elo_winner_surf - elo_loser_surf)

            # Probabilidad esperada: usa ELO de superficie, igual que en producción
            expected_prob = self.expected_score(elo_winner_surf, elo_loser_surf)
            expected_probs.append(expected_prob)

            # ACTUALIZAR ELOs para el siguiente partido
            # Calcular margen de sets (diferencia: sets_ganador - sets_perdedor)
            margin = 0
            if "score" in row and pd.notna(row["score"]):
                score_str = str(row["score"])
                try:
                    sets = score_str.split()
                    winner_sets = 0
                    loser_sets = 0
                    for s in sets:
                        # Ignorar tokens que no sean scores de set (RET, W/O, etc.)
                        m = re.match(r'^(\d+)-(\d+)', s)
                        if m:
                            a, b = int(m.group(1)), int(m.group(2))
                            if a > b:
                                winner_sets += 1
                            elif b > a:
                                loser_sets += 1
                    margin = winner_sets - loser_sets
                except Exception:
                    margin = 0

            self.update_ratings(winner, loser, surface, margin)

        # Añadir columnas al DataFrame
        df["winner_elo"] = winner_elos
        df["loser_elo"] = loser_elos
        df["winner_elo_surface"] = winner_elos_surface
        df["loser_elo_surface"] = loser_elos_surface
        df["elo_diff"] = elo_diffs
        df["elo_diff_surface"] = elo_diffs_surface
        df["elo_expected_prob"] = expected_probs

        logger.info(f"\n✅ ELOs calculados para {len(df):,} partidos")
        logger.info(f"📊 Jugadores únicos: {len(self.ratings):,}")
        logger.info(f"📊 ELO promedio: {np.mean(list(self.ratings.values())):.1f}")
        logger.info(f"📊 ELO std: {np.std(list(self.ratings.values())):.1f}")

        return df

    def get_current_ratings(self, top_n=20, surface=None):
        """
        Devuelve los top N jugadores por ELO actual

        Args:
            top_n: Número de jugadores a devolver
            surface: Si se especifica, devuelve ratings de esa superficie

        Returns:
            DataFrame con rankings
        """
        if surface and surface in self.ratings_by_surface:
            ratings_dict = self.ratings_by_surface[surface]
        else:
            ratings_dict = self.ratings

        sorted_ratings = sorted(ratings_dict.items(), key=lambda x: x[1], reverse=True)

        return pd.DataFrame(sorted_ratings[:top_n], columns=["Jugador", "ELO"])

    def get_player_elo_history(self, player_name, df_with_elo):
        """
        Obtiene el historial de ELO de un jugador

        Args:
            player_name: Nombre del jugador
            df_with_elo: DataFrame con columnas de ELO calculadas

        Returns:
            DataFrame con fechas y ELOs del jugador
        """
        # Partidos donde ganó
        winner_matches = df_with_elo[df_with_elo["winner_name"] == player_name][
            ["tourney_date", "winner_elo", "winner_elo_surface", "surface"]
        ].copy()
        winner_matches.columns = ["fecha", "elo", "elo_surface", "surface"]

        # Partidos donde perdió
        loser_matches = df_with_elo[df_with_elo["loser_name"] == player_name][
            ["tourney_date", "loser_elo", "loser_elo_surface", "surface"]
        ].copy()
        loser_matches.columns = ["fecha", "elo", "elo_surface", "surface"]

        # Combinar y ordenar
        history = pd.concat([winner_matches, loser_matches]).sort_values("fecha")

        return history


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Añadir path para imports
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # Cargar datos limpios
    logger.info("Cargando datos...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    # Crear sistema ELO
    elo_system = TennisELO(k_factor=32, base_rating=1500)

    # Calcular ELOs históricos
    df_con_elo = elo_system.calculate_historical_elos(df)

    # Ver top jugadores
    logger.info("\n" + "=" * 60)
    logger.info("🏆 TOP 20 JUGADORES POR ELO")
    logger.info("=" * 60)
    print(elo_system.get_current_ratings(top_n=20))

    # Ver top por superficie
    for superficie in ["Hard", "Clay", "Grass"]:
        logger.info(f"\n🎾 TOP 10 en {superficie}:")
        print(elo_system.get_current_ratings(top_n=10, surface=superficie))

    # Guardar dataset con ELO
    output_file = "datos/processed/atp_matches_con_elo.csv"
    df_con_elo.to_csv(output_file, index=False)
    logger.info(f"\n💾 Dataset con ELO guardado en: {output_file}")

    # Estadísticas de calibración
    logger.info("\n" + "=" * 60)
    logger.info("📊 CALIBRACIÓN DEL ELO")
    logger.info("=" * 60)

    # Accuracy del ELO (predicciones correctas)
    df_con_elo["elo_prediction"] = (df_con_elo["elo_expected_prob"] >= 0.5).astype(int)
    df_con_elo["actual"] = 1  # Siempre 1 porque winner ganó
    accuracy = (df_con_elo["elo_prediction"] == df_con_elo["actual"]).mean()

    logger.info(f"Accuracy del ELO: {accuracy*100:.2f}%")
    logger.info(f"Probabilidad promedio del favorito: {df_con_elo['elo_expected_prob'].mean():.3f}")

    logger.info("\n✅ Proceso completado!")
