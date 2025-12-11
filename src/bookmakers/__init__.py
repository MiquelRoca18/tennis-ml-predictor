"""
Bookmakers Module - Sistema de comparación de cuotas de múltiples casas de apuestas

Este módulo proporciona funcionalidad para:
- Obtener cuotas de múltiples bookmakers vía The Odds API
- Comparar cuotas y encontrar la mejor disponible
- Sistema de alertas para oportunidades de alto EV
- Integración con Kelly Criterion para optimización de apuestas
"""

from .odds_fetcher import OddsFetcher
from .odds_comparator import OddsComparator
from .alert_system import AlertSystem
from .config import BookmakerConfig

__all__ = [
    'OddsFetcher',
    'OddsComparator',
    'AlertSystem',
    'BookmakerConfig'
]
