"""
Módulo de Validación Walk-Forward
==================================

Wrapper para validación walk-forward (Fase 3).
"""

import sys
from pathlib import Path
from src.utils import print_header


def validar_walkforward():
    """Ejecuta validación walk-forward"""
    print_header("VALIDACIÓN WALK-FORWARD - FASE 3", "📊")
    
    # Añadir scripts al path
    scripts_path = Path(__file__).parent.parent.parent / "scripts" / "internal"
    sys.path.insert(0, str(scripts_path))
    
    try:
        # Importar y ejecutar el script de validación
        import walk_forward_validation
        walk_forward_validation.main()
        return True
    except ImportError as e:
        print(f"⚠️  Script de validación walk-forward no encontrado: {e}")
        print("   Ejecutar manualmente: python scripts/internal/walk_forward_validation.py")
        return False
    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return False
