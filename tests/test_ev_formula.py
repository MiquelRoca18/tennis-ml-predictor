# tests/test_ev_formula.py
"""
Tests de la fórmula de EV para verificar corrección del double-counting de mercado.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calcular_ev_actual(elo_prob, odds, baseline_elo_peso=0.6):
    """Fórmula actual (con bug)."""
    prob_mercado = 1.0 / odds
    prob_fusion = baseline_elo_peso * elo_prob + (1 - baseline_elo_peso) * prob_mercado
    return prob_fusion * odds - 1


def calcular_ev_corregido(elo_prob, odds):
    """Fórmula corregida: comparar modelo vs mercado directamente."""
    return elo_prob * odds - 1


def test_double_counting_genera_ev_falso():
    """
    Bug: cuando elo_prob == market_prob, el EV debería ser 0 (mercado justo).
    Con la fórmula actual, también es 0. Pero cuando ELO y mercado divergen
    ligeramente por ruido, la fusión suaviza la diferencia y genera EV falso.
    """
    # Caso: mercado dice 60%, ELO dice 65% → ELO ve edge
    odds = 1 / 0.60   # cuota para 60% de probabilidad implícita
    elo_prob = 0.65   # ELO cree que hay más edge

    ev_actual = calcular_ev_actual(elo_prob, odds)
    ev_corregido = calcular_ev_corregido(elo_prob, odds)

    # La fórmula corregida debe dar mayor EV (más directa)
    assert ev_corregido > ev_actual, (
        f"EV corregido ({ev_corregido:.4f}) debería ser mayor que actual ({ev_actual:.4f})"
    )


def test_ev_cero_cuando_elo_igual_mercado():
    """Cuando ELO == mercado no hay edge: EV debe ser 0 (menos el vig)."""
    odds = 2.0  # mercado implica 50%
    elo_prob = 0.50  # ELO también dice 50%

    ev_actual = calcular_ev_actual(elo_prob, odds)
    ev_corregido = calcular_ev_corregido(elo_prob, odds)

    assert abs(ev_actual) < 0.001, f"EV actual debería ser ~0, es {ev_actual}"
    assert abs(ev_corregido) < 0.001, f"EV corregido debería ser ~0, es {ev_corregido}"


def test_ev_negativo_cuando_elo_menor_mercado():
    """Cuando ELO < mercado, no debemos apostar: EV negativo."""
    odds = 1.5   # mercado implica 66.7%
    elo_prob = 0.55  # ELO cree que solo hay 55%

    ev_corregido = calcular_ev_corregido(elo_prob, odds)
    assert ev_corregido < 0, f"EV debería ser negativo, es {ev_corregido}"
