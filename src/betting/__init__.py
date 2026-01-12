"""
Módulo de Betting - Tennis ML Predictor
=======================================

Módulo centralizado para toda la lógica relacionada con apuestas:
- Kelly Criterion
- Simulación de bankroll
- Predictor con múltiples bookmakers

Uso:
    from src.betting import KellyCalculator, BankrollSimulator
    from src.betting.predictor import PredictorMultiBookmaker
"""

from .kelly_calculator import KellyCalculator
from .bankroll_simulator import BankrollSimulator

__all__ = ["KellyCalculator", "BankrollSimulator"]
