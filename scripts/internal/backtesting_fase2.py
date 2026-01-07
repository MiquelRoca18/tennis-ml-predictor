"""
Wrapper para ejecutar el backtesting de producción real desde el pipeline
Usa el backtesting completo con datos de 2024
"""
import sys
import subprocess
from pathlib import Path

def main():
    """
    Ejecuta el backtesting de producción real para 2024
    """
    print("="*70)
    print("🎲 EJECUTANDO BACKTESTING DE PRODUCCIÓN REAL")
    print("="*70)
    print("\nEste backtesting usa:")
    print("  - Datos reales de 2024")
    print("  - Cuotas históricas")
    print("  - Estrategia conservadora con Kelly Criterion")
    print("  - Predicción bidireccional")
    print("\n" + "="*70 + "\n")
    
    # Ejecutar el script de backtesting completo
    script_path = Path("scripts/backtesting_produccion_real_completo.py")
    
    if not script_path.exists():
        print(f"❌ Error: No se encuentra {script_path}")
        return 1
    
    # Ejecutar el script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path.cwd()
    )
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
