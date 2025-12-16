"""
Módulo de Configuración Centralizada
====================================

Este módulo proporciona acceso centralizado a toda la configuración del proyecto.

Uso:
    from src.config import Config
    
    # Acceder a configuración
    api_key = Config.ODDS_API_KEY
    model_path = Config.MODEL_PATH
"""

from .settings import Config, ENV_TEMPLATE

__all__ = ['Config', 'ENV_TEMPLATE']
