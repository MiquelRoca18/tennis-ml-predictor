"""
Módulo de Validación - Automatización
====================================

Wrapper optimizado para validación de automatización.
"""

import sys
from pathlib import Path

from src.utils import print_header


def validar_automatizacion():
    """Valida automatización (Fase 7)"""
    print_header("VALIDACIÓN AUTOMATIZACIÓN - FASE 7", "🤖")
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'deprecated'))
    
    from validacion_fase7 import main as validar_fase7
    validar_fase7()
    
    return True
