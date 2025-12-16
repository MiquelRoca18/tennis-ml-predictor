"""
Módulo de Demos - Bookmakers
============================

Demo refactorizada de múltiples bookmakers.
"""

from src.utils import print_header, print_metric
from src.config import Config
import sys
from pathlib import Path


def demo_bookmakers(modo_simulado=False):
    """Demo de múltiples bookmakers"""
    if modo_simulado:
        print_header("DEMO - BOOKMAKERS (SIMULADO)", "🌐")
    else:
        print_header("DEMO - BOOKMAKERS (REAL)", "🌐")
    
    try:
        # Por ahora, usar scripts existentes (son complejos)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'deprecated'))
        
        if modo_simulado:
            from demo_fase6_simulado import main as demo_main
        else:
            from demo_multibookmaker_fase6 import main as demo_main
        
        demo_main()
        
        print("\n✅ Demo de bookmakers completada!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
