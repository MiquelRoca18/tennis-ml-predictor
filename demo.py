#!/usr/bin/env python3
"""
Script Unificado de Demos - Tennis ML Predictor
===============================================

VERSIÓN OPTIMIZADA - Código refactorizado y limpio.

Uso:
    python demo.py --all                # Ejecuta todas las demos
    python demo.py --feature tracking   # Demo de tracking
    python demo.py --feature kelly      # Demo de Kelly Criterion
    python demo.py --feature bookmakers # Demo de múltiples bookmakers
    python demo.py --mode simulated     # Modo simulado (sin API real)
"""

import argparse
import sys
from pathlib import Path

# Añadir path para imports
sys.path.append(str(Path(__file__).parent))

from src.utils import print_header, print_summary, safe_execute
from src.demos import demo_tracking, demo_kelly, demo_bookmakers


def ejecutar_demo_tracking():
    """Ejecuta demo de tracking"""
    return safe_execute(demo_tracking, "Error en demo de tracking", verbose=True)


def ejecutar_demo_kelly():
    """Ejecuta demo de Kelly"""
    return safe_execute(demo_kelly, "Error en demo de Kelly", verbose=True)


def ejecutar_demo_bookmakers(modo_simulado=False):
    """Ejecuta demo de bookmakers"""
    def _demo():
        demo_bookmakers(modo_simulado=modo_simulado)
    
    return safe_execute(_demo, "Error en demo de bookmakers", verbose=True)


def demo_todas(modo_simulado=False):
    """Ejecuta todas las demos"""
    print_header("DEMOS COMPLETAS - TODAS LAS FEATURES", "🎯")
    
    resultados = {
        'Tracking System': ejecutar_demo_tracking(),
        'Kelly Criterion': ejecutar_demo_kelly(),
        'Múltiples Bookmakers': ejecutar_demo_bookmakers(modo_simulado=modo_simulado)
    }
    
    # Resumen usando utilidad compartida
    return print_summary(resultados, "RESUMEN DE DEMOS")


def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(
        description='Script unificado de demos para Tennis ML Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --all                    # Ejecuta todas las demos
  %(prog)s --feature tracking       # Demo de tracking
  %(prog)s --feature kelly          # Demo de Kelly Criterion
  %(prog)s --feature bookmakers     # Demo de múltiples bookmakers
  %(prog)s --mode simulated         # Modo simulado (sin API real)
  %(prog)s --all --mode simulated   # Todas las demos en modo simulado
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Ejecuta todas las demos')
    parser.add_argument('--feature', choices=['tracking', 'kelly', 'bookmakers'],
                       help='Ejecuta demo de una feature específica')
    parser.add_argument('--mode', choices=['real', 'simulated'], default='real',
                       help='Modo de ejecución (real o simulado, solo para bookmakers)')
    
    args = parser.parse_args()
    
    if not (args.all or args.feature):
        parser.print_help()
        sys.exit(1)
    
    # Determinar modo
    modo_simulado = (args.mode == 'simulated')
    
    # Ejecutar demos
    exito = True
    
    if args.all:
        exito = demo_todas(modo_simulado=modo_simulado)
    elif args.feature:
        if args.feature == 'tracking':
            exito = ejecutar_demo_tracking()
        elif args.feature == 'kelly':
            exito = ejecutar_demo_kelly()
        elif args.feature == 'bookmakers':
            exito = ejecutar_demo_bookmakers(modo_simulado=modo_simulado)
    
    sys.exit(0 if exito else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
