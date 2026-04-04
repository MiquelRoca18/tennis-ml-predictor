# scripts/comparar_modelos.py
"""
Comparación de modelos de predicción de tenis
==============================================

Ejecuta backtesting para 4 configuraciones y presenta tabla comparativa:
  1. Baseline original (0.6*ELO + 0.4*mercado)
  2. ELO puro (fórmula corregida, sin double-counting)
  3. WElo (K-factor variable + decay temporal)
  4. LightGBM (H2H + fatiga + forma + superficie) — si existe el modelo

Uso:
  python scripts/comparar_modelos.py
"""

import os
import sys
import subprocess
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPT_BACKTEST = "scripts/backtesting_produccion_real_completo.py"
LGBM_MODEL_PATH = "modelos/lgbm_tennis.pkl"

CONFIGURACIONES = [
    {
        "nombre": "1. Baseline (ELO+mercado original)",
        "env": {
            "BACKTEST_PRESET": "mejor",
            "BACKTEST_FLAT_STAKE": "1",
            "BACKTEST_FLAT_STAKE_EUR": "10",
        },
    },
    {
        "nombre": "2. ELO puro (fórmula corregida)",
        "env": {
            "BACKTEST_PRESET": "mejor",
            "BACKTEST_FLAT_STAKE": "1",
            "BACKTEST_FLAT_STAKE_EUR": "10",
            "BACKTEST_EV_FORMULA": "corregida",
        },
    },
    {
        "nombre": "3. WElo (K variable + decay)",
        "env": {
            "BACKTEST_PRESET": "mejor",
            "BACKTEST_FLAT_STAKE": "1",
            "BACKTEST_FLAT_STAKE_EUR": "10",
            "BACKTEST_EV_FORMULA": "corregida",
            "USE_WELO": "true",
        },
    },
]

if Path(LGBM_MODEL_PATH).exists():
    CONFIGURACIONES.append({
        "nombre": "4. LightGBM (H2H + fatiga + forma)",
        "env": {
            "BACKTEST_PRESET": "mejor",
            "BACKTEST_FLAT_STAKE": "1",
            "BACKTEST_FLAT_STAKE_EUR": "10",
            "BACKTEST_LGBM_MODEL": LGBM_MODEL_PATH,
        },
    })


def ejecutar_backtest(config: dict) -> str:
    """Ejecuta el backtesting con la configuración dada."""
    env = {**os.environ, **config["env"]}
    result = subprocess.run(
        [sys.executable, SCRIPT_BACKTEST],
        capture_output=True, text=True, env=env, timeout=300
    )
    return result.stdout + result.stderr


def extraer_roi_flat(output: str) -> dict:
    """
    Extrae ROI por año del output del backtesting.
    Busca líneas como "2024: 360 apuestas | ... | ROI +28.47%"
    """
    rois = {}
    patrones = [
        r"(\d{4}):\s+(\d+)\s+apuestas.*?ROI\s+([+-]?\d+\.?\d*)%",
        r"(\d{4}).*?(\d+)\s+apuestas.*?ROI\s+([+-]?\d+\.?\d*)%",
        r"(\d{4}).*?ROI\s+([+-]?\d+\.?\d*)%",
    ]
    for patron in patrones:
        for match in re.finditer(patron, output, re.IGNORECASE):
            año = int(match.group(1))
            roi = float(match.group(match.lastindex))
            if 2020 <= año <= 2025 and año not in rois:
                rois[año] = roi
        if rois:
            break
    return rois


def extraer_apuestas(output: str) -> dict:
    """Extrae número de apuestas por año."""
    apuestas = {}
    patron = r"(\d{4}):\s+(\d+)\s+apuestas"
    for match in re.finditer(patron, output, re.IGNORECASE):
        año = int(match.group(1))
        n = int(match.group(2))
        if 2020 <= año <= 2025:
            apuestas[año] = n
    return apuestas


def extraer_clv(output: str) -> tuple:
    """Extrae CLV medio y % CLV>0 del output."""
    match = re.search(r"CLV medio:\s+([+-]?\d+\.\d+).*?CLV>0:\s+(\d+\.?\d*)%", output)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def main():
    print("\n" + "=" * 72)
    print("COMPARACIÓN DE MODELOS — Tennis ML Predictor")
    print("=" * 72)

    todos_rois = {}
    todos_apuestas = {}
    todos_clv = {}

    for config in CONFIGURACIONES:
        nombre = config["nombre"]
        print(f"\n> {nombre}...")
        try:
            output = ejecutar_backtest(config)
            rois = extraer_roi_flat(output)
            apuestas = extraer_apuestas(output)
            clv_medio, clv_pct = extraer_clv(output)
            todos_rois[nombre] = rois
            todos_apuestas[nombre] = apuestas
            todos_clv[nombre] = (clv_medio, clv_pct)

            if rois:
                for año, roi in sorted(rois.items()):
                    n_ap = apuestas.get(año, "?")
                    print(f"   {año}: {n_ap} apuestas | {roi:+.1f}% ROI flat")
                if clv_medio is not None:
                    print(f"   CLV medio: {clv_medio:+.4f} | CLV>0: {clv_pct:.1f}%")
            else:
                print("   (No se pudo extraer ROI — revisar output)")
                for line in output.strip().split("\n")[-15:]:
                    if any(k in line.lower() for k in ["roi", "acierto", "apuesta", "año"]):
                        print(f"   LOG: {line.strip()}")
        except Exception as e:
            print(f"   Error: {e}")
            todos_rois[nombre] = {}
            todos_clv[nombre] = (None, None)

    # Tabla comparativa
    años = [2021, 2022, 2023, 2024]
    print("\n" + "=" * 72)
    print("TABLA COMPARATIVA — ROI Flat Stake (10 EUR por apuesta)")
    print("=" * 72)
    print(f"{'Modelo':<42} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'Media':>7}")
    print("-" * 72)

    for nombre, rois in todos_rois.items():
        vals = [rois.get(a) for a in años]
        roi_strs = [f"{v:+7.1f}" if v is not None else "    N/A" for v in vals]
        valid = [v for v in vals if v is not None]
        media = sum(valid) / len(valid) if valid else float("nan")
        media_str = f"{media:+7.1f}" if valid else "    N/A"
        print(f"{nombre:<42}{''.join(roi_strs)}{media_str}")

    # Tabla CLV
    print("\n" + "=" * 72)
    print("CLV (Closing Line Value) — edge real vs mercado")
    print("=" * 72)
    print(f"{'Modelo':<42} {'CLV medio':>12} {'CLV>0 %':>10}")
    print("-" * 66)
    for nombre, (clv_medio, clv_pct) in todos_clv.items():
        clv_str = f"{clv_medio:+.4f}" if clv_medio is not None else "N/A"
        pct_str = f"{clv_pct:.1f}%" if clv_pct is not None else "N/A"
        print(f"{nombre:<42} {clv_str:>12} {pct_str:>10}")

    print("\nComparación completada.")
    print("\nInterpretación:")
    print("  ROI flat >5% sostenido = edge genuino")
    print("  ROI flat <0% = el modelo no supera al mercado")
    print("  CLV medio >0 = modelo asigna mas probabilidad que el mercado (edge real)")
    print("  Mejor modelo = mayor ROI en 2024 (año mas reciente y dificil)")


if __name__ == "__main__":
    main()
