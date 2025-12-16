#!/usr/bin/env python3
"""
Script Unificado de Validación - Tennis ML Predictor
====================================================

VERSIÓN OPTIMIZADA - Código refactorizado y limpio.

Uso:
    python validate.py --all                    # Ejecuta todas las validaciones
    python validate.py --phase 2                # Valida Fase 2 (calibración)
    python validate.py --phase 3                # Valida Fase 3 (walk-forward)
    python validate.py --phase 5                # Valida Fase 5 (Kelly)
    python validate.py --phase 6                # Valida Fase 6 (bookmakers)
    python validate.py --phase 7                # Valida Fase 7 (automatización)
"""

import argparse
import sys
from pathlib import Path

# Añadir path para imports
sys.path.append(str(Path(__file__).parent))

from src.utils import print_header, print_summary, safe_execute
from src.validation import (
    validar_calibracion,
    validar_kelly,
    validar_walkforward,
    validar_bookmakers,
    validar_automatizacion
)


def validar_fase2_calibracion():
    """Valida calibración de modelos (Fase 2)"""
    print_header("VALIDACIÓN FASE 2 - CALIBRACIÓN")
    return safe_execute(validar_calibracion, "Error en validación de calibración", verbose=True)


def validar_fase3_walkforward():
    """Valida Walk-Forward y optimización (Fase 3)"""
    print_header("VALIDACIÓN FASE 3 - WALK-FORWARD")
    return safe_execute(validar_walkforward, "Error en validación Fase 3", verbose=True)


def validar_fase5_kelly_criterion():
    """Valida Kelly Criterion (Fase 5)"""
    print_header("VALIDACIÓN FASE 5 - KELLY CRITERION", "💎")
    return safe_execute(validar_kelly, "Error en validación Kelly", verbose=True)


def validar_fase6_bookmakers():
    """Valida múltiples bookmakers (Fase 6)"""
    print_header("VALIDACIÓN FASE 6 - MÚLTIPLES BOOKMAKERS", "🌐")
    return safe_execute(validar_bookmakers, "Error en validación Fase 6", verbose=True)


def validar_fase7_automatizacion():
    """Valida automatización (Fase 7)"""
    print_header("VALIDACIÓN FASE 7 - AUTOMATIZACIÓN", "🤖")
    return safe_execute(validar_automatizacion, "Error en validación Fase 7", verbose=True)


def validar_todas():
    """Ejecuta todas las validaciones"""
    print_header("VALIDACIÓN COMPLETA - TODAS LAS FASES", "🎯")
    
    resultados = {
        'Fase 2 - Calibración': validar_fase2_calibracion(),
        'Fase 3 - Walk-Forward': validar_fase3_walkforward(),
        'Fase 5 - Kelly Criterion': validar_fase5_kelly_criterion(),
        'Fase 6 - Bookmakers': validar_fase6_bookmakers(),
        'Fase 7 - Automatización': validar_fase7_automatizacion()
    }
    
    # Resumen usando utilidad compartida
    return print_summary(resultados, "RESUMEN DE VALIDACIONES")


def main():
    """Función principal con CLI"""
    parser = argparse.ArgumentParser(
        description='Script unificado de validación para Tennis ML Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --all                    # Ejecuta todas las validaciones
  %(prog)s --phase 2                # Valida Fase 2 (calibración)
  %(prog)s --phase 3                # Valida Fase 3 (walk-forward)
  %(prog)s --phase 5                # Valida Fase 5 (Kelly)
  %(prog)s --phase 6                # Valida Fase 6 (bookmakers)
  %(prog)s --phase 7                # Valida Fase 7 (automatización)
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Ejecuta todas las validaciones')
    parser.add_argument('--phase', type=int, choices=[2, 3, 5, 6, 7], help='Valida una fase específica')
    parser.add_argument('--component', choices=['calibration', 'walkforward', 'kelly', 'bookmakers', 'automation'],
                       help='Valida un componente específico')
    
    args = parser.parse_args()
    
    if not (args.all or args.phase or args.component):
        parser.print_help()
        sys.exit(1)
    
    # Ejecutar validaciones
    exito = True
    
    if args.all:
        exito = validar_todas()
    elif args.phase:
        if args.phase == 2:
            exito = validar_fase2_calibracion()
        elif args.phase == 3:
            exito = validar_fase3_walkforward()
        elif args.phase == 5:
            exito = validar_fase5_kelly_criterion()
        elif args.phase == 6:
            exito = validar_fase6_bookmakers()
        elif args.phase == 7:
            exito = validar_fase7_automatizacion()
    elif args.component:
        if args.component == 'calibration':
            exito = validar_fase2_calibracion()
        elif args.component == 'walkforward':
            exito = validar_fase3_walkforward()
        elif args.component == 'kelly':
            exito = validar_fase5_kelly_criterion()
        elif args.component == 'bookmakers':
            exito = validar_fase6_bookmakers()
        elif args.component == 'automation':
            exito = validar_fase7_automatizacion()
    
    sys.exit(0 if exito else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error en validación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
