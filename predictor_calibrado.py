"""
DEPRECATED: Este archivo se mantiene solo para compatibilidad hacia atrás.

Usar el módulo prediction en su lugar:
    from src.prediction import PredictorCalibrado

Este archivo ahora es solo un wrapper.
"""

import warnings

warnings.warn(
    "predictor_calibrado está deprecated. "
    "Usar 'from src.prediction import PredictorCalibrado' en su lugar.",
    DeprecationWarning,
    stacklevel=2
)

from src.prediction.predictor_calibrado import PredictorCalibrado

__all__ = ['PredictorCalibrado']
