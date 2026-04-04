# tests/test_welo.py
"""
Tests del sistema WElo (Weighted ELO).
K-factor variable por tourney_level. Grand Slam actualiza más que ATP250.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.welo_rating_system import WEloRatingSystem


def test_grand_slam_k_factor_mayor_que_atp250():
    """Los partidos de Grand Slam deben mover más el ELO que ATP250."""
    welo_gs = WEloRatingSystem()
    welo_atp250 = WEloRatingSystem()

    welo_gs.update_ratings("Alcaraz", "Sinner", "Hard", tourney_level="Grand Slam")
    welo_atp250.update_ratings("Alcaraz", "Sinner", "Hard", tourney_level="ATP250")

    elo_gs = welo_gs.get_rating("Alcaraz")
    elo_atp250 = welo_atp250.get_rating("Alcaraz")

    assert elo_gs > elo_atp250, (
        f"Grand Slam ELO ({elo_gs:.1f}) debería ser > ATP250 ELO ({elo_atp250:.1f})"
    )


def test_rating_inicial_es_1500():
    """Jugador nuevo comienza en 1500."""
    welo = WEloRatingSystem()
    assert welo.get_rating("Nuevo Jugador") == 1500.0


def test_ganador_sube_perdedor_baja():
    """El ganador sube y el perdedor baja simétricamente."""
    welo = WEloRatingSystem(base_rating=1500)
    welo.update_ratings("A", "B", "Hard", tourney_level="ATP500")

    rating_a = welo.get_rating("A")
    rating_b = welo.get_rating("B")

    assert rating_a > 1500, "El ganador debe subir de 1500"
    assert rating_b < 1500, "El perdedor debe bajar de 1500"
    assert abs((rating_a - 1500) - (1500 - rating_b)) < 0.01, "Cambios simétricos"


def test_expected_score_entre_0_y_1():
    """La probabilidad esperada debe estar entre 0 y 1."""
    welo = WEloRatingSystem()
    prob = welo.expected_score(1600, 1400)
    assert 0 < prob < 1
    assert prob > 0.5


def test_decay_temporal_reduce_rating_viejo():
    """Un partido de hace 3 años debe tener menos impacto que uno reciente."""
    import pandas as pd

    welo_con_decay = WEloRatingSystem(use_temporal_decay=True)
    welo_sin_decay = WEloRatingSystem(use_temporal_decay=False)

    df_historico = pd.DataFrame([
        {
            "tourney_date": pd.Timestamp("2021-01-01"),
            "winner_name": "PlayerA",
            "loser_name": "PlayerB",
            "surface": "Hard",
            "tourney_level": "ATP250",
        },
        {
            "tourney_date": pd.Timestamp("2024-01-01"),
            "winner_name": "PlayerB",
            "loser_name": "PlayerA",
            "surface": "Hard",
            "tourney_level": "ATP250",
        },
    ])

    welo_con_decay.calculate_historical_elos(df_historico)
    welo_sin_decay.calculate_historical_elos(df_historico)

    rating_decay = welo_con_decay.get_rating("PlayerA")
    rating_nodecay = welo_sin_decay.get_rating("PlayerA")

    assert rating_decay != rating_nodecay, "Decay debe producir distinto ELO"
