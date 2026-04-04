# scripts/train_model_lgbm.py
"""
Pipeline de entrenamiento LightGBM — Walk-forward real
=======================================================

Proceso:
1. Carga datos de odds (2021-2024) con cuotas
2. Carga partidos TML históricos (2018-2024)
3. Para cada partido en odds, calcula features usando solo datos PREVIOS
4. Walk-forward: entrena con datos < año Y, evalúa en año Y
5. Guarda modelo final como modelos/lgbm_tennis.pkl

Features (NO contienen cuota — evitar double-counting):
  elo_diff_surface, h2h_reciente_rate, diff_win_rate_60d,
  diff_carga_score, diff_dias_descanso, ventaja_superficie,
  rank_diff, elo_diff

Uso:
  python scripts/train_model_lgbm.py
  python scripts/train_model_lgbm.py --years 2022 2023 2024 --output modelos/lgbm_tennis.pkl
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.elo_rating_system import TennisELO
from src.features.features_h2h_mejorado import HeadToHeadCalculator
from src.features.features_forma_reciente import FormaRecienteCalculator
from src.features.features_fatiga import FatigaCalculator
from src.features.features_superficie import SuperficieSpecializationCalculator
from src.utils.player_name_normalizer import PlayerNameNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "elo_diff_surface",
    "h2h_reciente_rate",
    "diff_win_rate_60d",
    "diff_carga_score",
    "diff_dias_descanso",
    "ventaja_superficie",
    "rank_diff",
    "elo_diff",
]

ODDS_DIR = Path("datos/odds_historicas")
TML_DIR = Path("datos/raw")
MODELOS_DIR = Path("modelos")
AÑOS_BACKTEST = [2022, 2023, 2024]
AÑOS_HISTORICO_INICIO = 2018


def cargar_odds(año: int) -> pd.DataFrame:
    """Carga el CSV de odds para un año dado."""
    candidates = [
        ODDS_DIR / f"tennis_odds_{año}_{año}.csv",
    ]
    # también buscar cualquier CSV con ese año en el nombre
    for c in ODDS_DIR.glob(f"*{año}*.csv"):
        if c not in candidates:
            candidates.append(c)

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha"])
            # Filtrar solo al año pedido (algunos CSV tienen múltiples años)
            df = df[df["fecha"].dt.year == año].copy()
            if len(df) > 0:
                return df
    raise FileNotFoundError(f"No se encontró archivo de odds para {año} en {ODDS_DIR}")


def cargar_tml_historico(hasta_año: int) -> pd.DataFrame:
    """Carga partidos TML desde AÑOS_HISTORICO_INICIO hasta (exclusive) hasta_año."""
    dfs = []
    for año in range(AÑOS_HISTORICO_INICIO, hasta_año):
        for candidate in [
            TML_DIR / f"atp_matches_{año}_tml.csv",
            TML_DIR / f"{año}.csv",
        ]:
            if candidate.exists():
                try:
                    df_a = pd.read_csv(candidate)
                    df_a["tourney_date"] = pd.to_datetime(df_a["tourney_date"], errors="coerce")
                    df_a = df_a.dropna(subset=["tourney_date", "winner_name", "loser_name"])
                    dfs.append(df_a)
                    break
                except Exception as e:
                    logger.warning(f"Error cargando {candidate}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos TML en {TML_DIR}")
    df = pd.concat(dfs, ignore_index=True).sort_values("tourney_date").reset_index(drop=True)
    logger.info(f"   Histórico TML: {len(df)} partidos (hasta {hasta_año - 1})")
    return df


def split_walk_forward(df: pd.DataFrame, test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide en train (< test_year) y test (== test_year) sin data leakage."""
    train = df[df["fecha"].dt.year < test_year].copy()
    test = df[df["fecha"].dt.year == test_year].copy()
    return train, test


def construir_features_partido(
    j1: str,
    j2: str,
    superficie: str,
    fecha: pd.Timestamp,
    df_historico_pre: pd.DataFrame,
    elo_system: TennisELO,
) -> dict:
    """Calcula features para un partido usando solo datos anteriores a fecha."""
    h2h_calc = HeadToHeadCalculator(df_historico_pre)
    forma_calc = FormaRecienteCalculator(df_historico_pre)
    fatiga_calc = FatigaCalculator(df_historico_pre)
    sup_calc = SuperficieSpecializationCalculator(df_historico_pre)

    sup_norm = sup_calc._normalizar_superficie(superficie)

    elo_j1 = elo_system.get_rating(j1, sup_norm)
    elo_j2 = elo_system.get_rating(j2, sup_norm)
    elo_j1_g = elo_system.get_rating(j1)
    elo_j2_g = elo_system.get_rating(j2)

    h2h = h2h_calc.calcular_h2h(j1, j2, fecha, sup_norm)
    forma_j1 = forma_calc.calcular_forma(j1, fecha)
    forma_j2 = forma_calc.calcular_forma(j2, fecha)
    fat_j1 = fatiga_calc.calcular_fatiga(j1, fecha)
    fat_j2 = fatiga_calc.calcular_fatiga(j2, fecha)
    ventaja = sup_calc.calcular_ventaja_superficie(j1, j2, fecha, sup_norm)

    return {
        "elo_diff_surface": float(elo_j1 - elo_j2),
        "elo_diff": float(elo_j1_g - elo_j2_g),
        "h2h_reciente_rate": float(h2h.get("h2h_reciente_rate", 0.5)),
        "diff_win_rate_60d": float(
            forma_j1.get("win_rate_60d", 0.5) - forma_j2.get("win_rate_60d", 0.5)
        ),
        "diff_carga_score": float(
            fat_j1.get("carga_reciente_score", 0.3) - fat_j2.get("carga_reciente_score", 0.3)
        ),
        "diff_dias_descanso": float(
            fat_j1.get("dias_desde_ultimo_partido", 7) - fat_j2.get("dias_desde_ultimo_partido", 7)
        ),
        "ventaja_superficie": float(ventaja.get("ventaja_superficie", 0.0)),
        "rank_diff": 0.0,  # se rellena desde los datos de odds
    }


def construir_dataset(
    df_odds: pd.DataFrame,
    df_tml_historico: pd.DataFrame,
    año_test: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Construye dataset para un año. Aplica randomización J1/J2 y calcula features
    usando solo datos previos a cada partido (sin data leakage).
    """
    logger.info(f"   Construyendo dataset para año {año_test}...")

    df_odds_año = df_odds[df_odds["fecha"].dt.year == año_test].copy()
    df_odds_año = df_odds_año.sort_values("fecha").reset_index(drop=True)

    if df_odds_año.empty:
        logger.warning(f"   Sin datos de odds para {año_test}")
        return pd.DataFrame()

    # Randomizar J1/J2 (mismo proceso que backtesting — elimina bias jugador_1=ganador)
    rng = np.random.default_rng(seed=año_test)
    swap_mask = rng.integers(0, 2, size=len(df_odds_año)).astype(bool)
    df_odds_año["ganador_es_j1"] = ~swap_mask

    swap_idx = df_odds_año.index[swap_mask]
    df_odds_año.loc[swap_idx, ["jugador_1", "jugador_2"]] = (
        df_odds_año.loc[swap_idx, ["jugador_2", "jugador_1"]].values
    )
    df_odds_año.loc[swap_idx, ["cuota_jugador_1", "cuota_jugador_2"]] = (
        df_odds_año.loc[swap_idx, ["cuota_jugador_2", "cuota_jugador_1"]].values
    )

    # ELO inicializado con datos hasta inicio del año de test
    df_hist_pre = df_tml_historico[
        df_tml_historico["tourney_date"] < pd.Timestamp(f"{año_test}-01-01")
    ].copy()

    elo_system = TennisELO(k_factor=32, base_rating=1500)
    elo_system.calculate_historical_elos(df_hist_pre)

    known_players = set(
        df_hist_pre["winner_name"].tolist() + df_hist_pre["loser_name"].tolist()
    )
    normalizer = PlayerNameNormalizer(known_players)

    rows = []
    total = len(df_odds_año)

    for i, (_, partido) in enumerate(df_odds_año.iterrows()):
        if i % 50 == 0:
            logger.info(f"      {i}/{total}...")

        j1_raw = str(partido.get("jugador_1", ""))
        j2_raw = str(partido.get("jugador_2", ""))
        j1 = normalizer.normalize(j1_raw)
        j2 = normalizer.normalize(j2_raw)
        superficie = str(partido.get("superficie", "Hard"))
        fecha = pd.to_datetime(partido["fecha"])
        ganador_es_j1 = bool(partido["ganador_es_j1"])

        df_hist_hasta_hoy = df_tml_historico[
            df_tml_historico["tourney_date"] < fecha
        ].copy()

        if df_hist_hasta_hoy.empty:
            continue

        try:
            features = construir_features_partido(
                j1, j2, superficie, fecha, df_hist_hasta_hoy, elo_system
            )
        except Exception as e:
            logger.debug(f"Error features {j1} vs {j2}: {e}")
            continue

        # rank_diff desde datos de odds
        rank_j1 = float(partido.get("ganador_rank") or 999)
        rank_j2 = float(partido.get("perdedor_rank") or 999)
        if not ganador_es_j1:
            rank_j1, rank_j2 = rank_j2, rank_j1
        features["rank_diff"] = rank_j2 - rank_j1

        features["target"] = int(ganador_es_j1)
        features["fecha"] = fecha
        features["cuota_j1"] = float(partido.get("cuota_jugador_1") or 0)
        features["cuota_j2"] = float(partido.get("cuota_jugador_2") or 0)
        features["año"] = año_test
        rows.append(features)

        # Actualizar ELO con resultado real
        winner = j1 if ganador_es_j1 else j2
        loser = j2 if ganador_es_j1 else j1
        elo_system.update_ratings(winner, loser, superficie)

    df_result = pd.DataFrame(rows)
    logger.info(f"   Dataset {año_test}: {len(df_result)} partidos con features")
    return df_result


def train_lgbm_model(df_train: pd.DataFrame, feature_cols: list):
    """Entrena LGBMClassifier calibrado. Devuelve el modelo."""
    X = df_train[feature_cols].fillna(0).values
    y = df_train["target"].values

    base_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    calibrated = CalibratedClassifierCV(base_model, cv=3, method="sigmoid")
    calibrated.fit(X, y)

    auc = roc_auc_score(y, calibrated.predict_proba(X)[:, 1])
    logger.info(f"   Modelo entrenado: {len(X)} muestras, AUC={auc:.4f}")
    return calibrated


def evaluar_modelo(model, df_test: pd.DataFrame, feature_cols: list) -> dict:
    """Evalúa el modelo. Devuelve métricas clave."""
    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test["target"].values

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    ganancias = []
    for prob, target, cuota in zip(probs, y_test, df_test["cuota_j1"]):
        if not cuota or cuota <= 0:
            continue
        market_prob = 1.0 / cuota
        if prob > market_prob:
            ganancia = float(cuota - 1) if target == 1 else -1.0
            ganancias.append(ganancia)

    roi = sum(ganancias) / max(len(ganancias), 1) * 100 if ganancias else 0.0
    win_rate = sum(1 for g in ganancias if g > 0) / max(len(ganancias), 1) * 100 if ganancias else 0.0

    return {
        "auc": auc,
        "brier": brier,
        "n_apuestas": len(ganancias),
        "win_rate_pct": win_rate,
        "roi_flat_pct": roi,
    }


def save_model(model, path: str) -> None:
    """Guarda el modelo en disco."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"   Modelo guardado: {path}")


def load_model(path: str):
    """Carga modelo desde disco."""
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Entrena LightGBM walk-forward para tenis")
    parser.add_argument("--years", type=int, nargs="+", default=AÑOS_BACKTEST)
    parser.add_argument("--output", type=str, default=str(MODELOS_DIR / "lgbm_tennis.pkl"))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO LGBM — Walk-forward Tennis Prediction")
    logger.info("=" * 60)

    # Cargar todos los odds disponibles
    all_odds = []
    for año in range(2021, 2025):
        try:
            df_o = cargar_odds(año)
            all_odds.append(df_o)
            logger.info(f"   Odds {año}: {len(df_o)} partidos")
        except FileNotFoundError as e:
            logger.warning(f"   {e}")
    if not all_odds:
        raise RuntimeError("No se encontraron datos de odds")
    df_odds_all = pd.concat(all_odds, ignore_index=True)

    max_year = max(args.years) + 1
    df_tml = cargar_tml_historico(hasta_año=max_year)

    resultados = []
    all_datasets = []

    for año_test in args.years:
        logger.info(f"\n{'='*40}")
        logger.info(f"WALK-FORWARD: Test {año_test}")
        logger.info(f"{'='*40}")

        df_test_ds = construir_dataset(df_odds_all, df_tml, año_test)
        if df_test_ds.empty:
            continue

        años_train = list(range(2021, año_test))
        df_train_parts = []
        for año_train in años_train:
            df_t = construir_dataset(df_odds_all, df_tml, año_train)
            if not df_t.empty:
                df_train_parts.append(df_t)
                all_datasets.append(df_t)

        if not df_train_parts:
            logger.warning(f"   Sin datos de entrenamiento para {año_test}")
            continue

        df_train_ds = pd.concat(df_train_parts, ignore_index=True)
        logger.info(f"   Train: {len(df_train_ds)} | Test: {len(df_test_ds)}")

        modelo = train_lgbm_model(df_train_ds, FEATURE_COLS)
        metricas = evaluar_modelo(modelo, df_test_ds, FEATURE_COLS)

        logger.info(
            f"   AUC={metricas['auc']:.4f} | "
            f"Brier={metricas['brier']:.4f} | "
            f"Apuestas={metricas['n_apuestas']} | "
            f"Win%={metricas['win_rate_pct']:.1f}% | "
            f"ROI={metricas['roi_flat_pct']:.1f}%"
        )
        resultados.append({"año": año_test, **metricas})

    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN WALK-FORWARD LGBM")
    logger.info("=" * 60)
    for r in resultados:
        logger.info(
            f"  {r['año']}: AUC={r['auc']:.4f} | "
            f"ROI={r['roi_flat_pct']:.1f}% | "
            f"Apuestas={r['n_apuestas']}"
        )
    if resultados:
        roi_medio = np.mean([r["roi_flat_pct"] for r in resultados])
        logger.info(f"  ROI medio: {roi_medio:.1f}%")

    # Modelo final con todos los datos
    if all_datasets:
        logger.info(f"\nEntrenando modelo final con {sum(len(d) for d in all_datasets)} partidos...")
        df_all = pd.concat(all_datasets, ignore_index=True)
        modelo_final = train_lgbm_model(df_all, FEATURE_COLS)
        save_model(modelo_final, args.output)
        logger.info(f"\n✅ Modelo final: {args.output}")
    else:
        logger.error("❌ Sin datos para modelo final")


if __name__ == "__main__":
    main()
