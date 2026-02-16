#!/usr/bin/env python3
"""
Procesa Excel de Tennis-Data.co.uk a CSV para backtesting
=========================================================

Convierte el formato Excel (Winner, Loser, B365W, B365L, etc.)
al formato esperado por backtesting_produccion_real_completo:
fecha, jugador_1, jugador_2, cuota_jugador_1, cuota_jugador_2, superficie, etc.

Uso:
    python scripts/internal/process_tennis_data_odds.py [año]
    # Si no se pasa año, procesa todos los .xlsx en datos/odds_historicas/
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ODDS_DIR = PROJECT_ROOT / "datos" / "odds_historicas"


def procesar_excel_a_csv(año: int) -> Path:
    """
    Procesa {año}.xlsx y genera tennis_odds_{año}_{año}.csv

    Returns:
        Path al CSV generado
    """
    excel_path = ODDS_DIR / f"{año}.xlsx"
    csv_path = ODDS_DIR / f"tennis_odds_{año}_{año}.csv"

    if not excel_path.exists():
        raise FileNotFoundError(f"No existe {excel_path}")

    df = pd.read_excel(excel_path)

    # Mapear columnas tennis-data.co.uk -> nuestro formato
    # Winner/Loser = jugadores (ganador es jugador_1 en el CSV de ejemplo)
    # B365W/B365L = cuotas. Fallback: PSW/PSL, AvgW/AvgL
    col_odds_w = "B365W" if "B365W" in df.columns else ("PSW" if "PSW" in df.columns else "AvgW")
    col_odds_l = "B365L" if "B365L" in df.columns else ("PSL" if "PSL" in df.columns else "AvgL")

    df_out = pd.DataFrame({
        "fecha": pd.to_datetime(df["Date"], errors="coerce"),
        "jugador_1": df["Winner"].astype(str),
        "jugador_2": df["Loser"].astype(str),
        "cuota_jugador_1": pd.to_numeric(df[col_odds_w], errors="coerce"),
        "cuota_jugador_2": pd.to_numeric(df[col_odds_l], errors="coerce"),
        "superficie": df["Surface"].fillna("Hard").astype(str),
        "bookmaker": "B365" if "B365W" in df.columns else "PS",
        "torneo": df["Tournament"].fillna("").astype(str),
        "serie": df["Series"].fillna("").astype(str),
        "ronda": df["Round"].fillna("").astype(str),
        "ganador_rank": pd.to_numeric(df["WRank"], errors="coerce").fillna(999),
        "perdedor_rank": pd.to_numeric(df["LRank"], errors="coerce").fillna(999),
    })

    # Eliminar filas sin cuotas válidas
    df_out = df_out.dropna(subset=["cuota_jugador_1", "cuota_jugador_2", "fecha"])
    df_out = df_out[(df_out["cuota_jugador_1"] > 1) & (df_out["cuota_jugador_2"] > 1)]

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_path, index=False)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Procesa Excel de Tennis-Data a CSV")
    parser.add_argument("año", type=int, nargs="?", help="Año a procesar (ej: 2022)")
    args = parser.parse_args()

    if args.año:
        años = [args.año]
    else:
        # Procesar todos los xlsx en el directorio
        años = []
        for f in ODDS_DIR.glob("*.xlsx"):
            try:
                años.append(int(f.stem))
            except ValueError:
                pass
        años = sorted(set(años))

    if not años:
        print("❌ No se encontraron archivos .xlsx en datos/odds_historicas/")
        print("   Usa: python scripts/internal/process_tennis_data_odds.py 2022")
        sys.exit(1)

    for año in años:
        try:
            csv_path = procesar_excel_a_csv(año)
            print(f"✅ {año}: {len(pd.read_csv(csv_path))} partidos → {csv_path.name}")
        except Exception as e:
            print(f"❌ {año}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
