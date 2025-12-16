"""
Módulo de Validación - Bookmakers
=================================

Wrapper optimizado para validación de bookmakers.
"""

import sys
from pathlib import Path

from src.utils import print_header


def validar_bookmakers():
    """Valida múltiples bookmakers (Fase 6)"""
    print_header("VALIDACIÓN BOOKMAKERS - FASE 6", "🌐")
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'deprecated'))
    
    from validacion_fase6 import main as validar_fase6
    validar_fase6()
    
    return True
