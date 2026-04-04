# src/features/welo_rating_system.py
"""
Weighted ELO Rating System (WElo)
==================================
Mejora sobre ELO estándar:
1. K-factor variable por nivel de torneo (Grand Slam > Masters > ATP500 > ATP250)
2. Decay temporal: partidos viejos pesan menos
3. ELO por superficie independiente (igual que TennisELO)

Referencia: Kovalchik (2021) "Weighted Elo rating for tennis match predictions"
DOI: 10.1016/j.ejor.2021.03.008 → ~3.56% ROI en 2012-2020
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

K_FACTORS = {
    "grand slam": 48,
    "grandslam": 48,
    "masters 1000": 36,
    "masters1000": 36,
    "masters cup": 36,
    "atp1000": 36,
    "1000": 36,
    "atp 500": 24,
    "atp500": 24,
    "500": 24,
    "atp 250": 16,
    "atp250": 16,
    "250": 16,
    "default": 24,
}

SURFACES = ["Hard", "Clay", "Grass"]


class WEloRatingSystem:
    """
    Sistema WElo: ELO ponderado por nivel de torneo y con decay temporal.
    API compatible con TennisELO: get_rating(), expected_score(), update_ratings(),
    calculate_historical_elos().
    """

    def __init__(
        self,
        base_rating: float = 1500.0,
        use_temporal_decay: bool = True,
        decay_half_life_days: int = 365,
    ):
        self.base_rating = base_rating
        self.use_temporal_decay = use_temporal_decay
        self.decay_half_life_days = decay_half_life_days

        self.ratings: dict = {}
        self.surface_ratings: dict = {s: {} for s in SURFACES}
        self._reference_date = None

    def get_rating(self, player: str, surface: str = None) -> float:
        """Devuelve el ELO del jugador (global o por superficie)."""
        if surface and surface in self.surface_ratings:
            return self.surface_ratings[surface].get(player, self.base_rating)
        return self.ratings.get(player, self.base_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probabilidad esperada de victoria de A contra B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _get_k_factor(self, tourney_level) -> float:
        if not tourney_level:
            return K_FACTORS["default"]
        key = str(tourney_level).lower().strip()
        return K_FACTORS.get(key, K_FACTORS["default"])

    def _decay_weight(self, match_date) -> float:
        if not self.use_temporal_decay or self._reference_date is None:
            return 1.0
        days_ago = (self._reference_date - pd.Timestamp(match_date)).days
        if days_ago <= 0:
            return 1.0
        return 2.0 ** (-days_ago / self.decay_half_life_days)

    def _normalize_surface(self, surface) -> str:
        if not surface:
            return "Hard"
        mapping = {
            "outdoor": "Hard", "indoor": "Hard", "carpet": "Hard",
            "hard": "Hard", "clay": "Clay", "grass": "Grass",
        }
        return mapping.get(str(surface).lower().strip(), "Hard")

    def update_ratings(
        self,
        winner: str,
        loser: str,
        surface: str,
        tourney_level=None,
        match_date=None,
    ) -> None:
        """Actualiza ratings globales y de superficie tras un partido."""
        surface = self._normalize_surface(surface)
        k = self._get_k_factor(tourney_level)

        if match_date is not None and self.use_temporal_decay:
            k = k * self._decay_weight(match_date)

        # Rating global
        ra = self.ratings.get(winner, self.base_rating)
        rb = self.ratings.get(loser, self.base_rating)
        ea = self.expected_score(ra, rb)
        self.ratings[winner] = ra + k * (1.0 - ea)
        self.ratings[loser] = rb + k * (0.0 - (1.0 - ea))

        # Rating por superficie
        surf = self.surface_ratings[surface]
        ra_s = surf.get(winner, self.base_rating)
        rb_s = surf.get(loser, self.base_rating)
        ea_s = self.expected_score(ra_s, rb_s)
        surf[winner] = ra_s + k * (1.0 - ea_s)
        surf[loser] = rb_s + k * (0.0 - (1.0 - ea_s))

    def calculate_historical_elos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalcula ELOs sobre un DataFrame histórico ordenado por fecha.
        Columnas requeridas: tourney_date, winner_name, loser_name, surface
        Columnas opcionales: tourney_level
        """
        df = df.copy().sort_values("tourney_date").reset_index(drop=True)

        if self.use_temporal_decay and len(df) > 0:
            self._reference_date = pd.to_datetime(df["tourney_date"].max())

        has_level = "tourney_level" in df.columns
        winner_elo_pre, loser_elo_pre = [], []
        winner_elo_surf_pre, loser_elo_surf_pre = [], []

        for _, row in df.iterrows():
            winner = row["winner_name"]
            loser = row["loser_name"]
            surface = self._normalize_surface(row.get("surface", "Hard"))
            level = str(row.get("tourney_level", "")).strip() if has_level else None
            date = pd.to_datetime(row["tourney_date"])

            winner_elo_pre.append(self.ratings.get(winner, self.base_rating))
            loser_elo_pre.append(self.ratings.get(loser, self.base_rating))
            winner_elo_surf_pre.append(self.surface_ratings[surface].get(winner, self.base_rating))
            loser_elo_surf_pre.append(self.surface_ratings[surface].get(loser, self.base_rating))

            self.update_ratings(winner, loser, surface, level, date)

        df["winner_welo_pre"] = winner_elo_pre
        df["loser_welo_pre"] = loser_elo_pre
        df["winner_welo_surface_pre"] = winner_elo_surf_pre
        df["loser_welo_surface_pre"] = loser_elo_surf_pre
        df["winner_welo_prob"] = [
            self.expected_score(w, l)
            for w, l in zip(winner_elo_surf_pre, loser_elo_surf_pre)
        ]

        logger.info(
            f"WElo calculado: {len(self.ratings)} jugadores, "
            f"decay={'on' if self.use_temporal_decay else 'off'}"
        )
        return df
