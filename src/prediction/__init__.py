"""
Módulo de Prediction - Tennis ML Predictor
==========================================

Módulo centralizado para toda la lógica de predicción:
- Predictor calibrado base
- Predictor con múltiples bookmakers

Uso:
    from src.prediction import PredictorCalibrado
"""

from .predictor_calibrado import PredictorCalibrado

__all__ = ["PredictorCalibrado"]
