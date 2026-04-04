# tests/test_lgbm_pipeline.py
"""
Tests del pipeline de entrenamiento LightGBM.
Verifica: sin data leakage, features correctas, calibración, formato de salida.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_fake_match_data(n=200, seed=42):
    """Genera datos de partidos sintéticos para tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="2D")
    return pd.DataFrame({
        "fecha": dates,
        "elo_diff": rng.normal(0, 150, n),
        "elo_diff_surface": rng.normal(0, 120, n),
        "h2h_reciente_rate": rng.uniform(0.3, 0.7, n),
        "diff_win_rate_60d": rng.normal(0, 0.15, n),
        "diff_carga_score": rng.normal(0, 0.3, n),
        "diff_dias_descanso": rng.normal(0, 3, n),
        "ventaja_superficie": rng.normal(0, 0.1, n),
        "rank_diff": rng.normal(0, 50, n),
        "cuota_j1": rng.uniform(1.3, 3.5, n),
        "target": rng.integers(0, 2, n),
    })


def test_features_correctas_en_dataframe():
    """El dataset de entrenamiento debe tener todas las features esperadas."""
    from scripts.train_model_lgbm import FEATURE_COLS
    df = make_fake_match_data()
    for feat in FEATURE_COLS:
        assert feat in df.columns, f"Feature '{feat}' falta en el DataFrame"


def test_sin_data_leakage_temporal():
    """El modelo para año Y nunca debe entrenarse con datos de año Y o posterior."""
    from scripts.train_model_lgbm import split_walk_forward

    # Construir datos que cubren 2020-2023 para garantizar datos en ambos splits
    dates_2020_2021 = pd.date_range("2020-01-01", "2021-12-31", freq="3D")
    dates_2022_2023 = pd.date_range("2022-01-01", "2023-12-31", freq="3D")
    all_dates = dates_2020_2021.append(dates_2022_2023)
    rng = np.random.default_rng(42)
    n = len(all_dates)
    df = pd.DataFrame({
        "fecha": all_dates,
        "elo_diff": rng.normal(0, 150, n),
        "target": rng.integers(0, 2, n),
    })

    train_df, test_df = split_walk_forward(df, test_year=2022)

    max_train_date = train_df["fecha"].max()
    min_test_date = test_df["fecha"].min()

    assert max_train_date < pd.Timestamp("2022-01-01"), (
        f"Train tiene datos de 2022+: {max_train_date}"
    )
    assert min_test_date >= pd.Timestamp("2022-01-01"), (
        f"Test tiene datos anteriores a 2022: {min_test_date}"
    )


def test_modelo_produce_probabilidades_validas():
    """El modelo entrenado debe producir probabilidades entre 0 y 1."""
    from scripts.train_model_lgbm import FEATURE_COLS, train_lgbm_model

    df = make_fake_match_data(n=300)
    model = train_lgbm_model(df, FEATURE_COLS)

    X = df[FEATURE_COLS].values
    probs = model.predict_proba(X)[:, 1]

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    assert len(probs) == len(df)


def test_modelo_guarda_y_carga_correctamente(tmp_path):
    """El modelo debe poder guardarse y cargarse sin perder capacidad predictiva."""
    from scripts.train_model_lgbm import FEATURE_COLS, train_lgbm_model, save_model, load_model

    df = make_fake_match_data(n=300)
    model_path = str(tmp_path / "test_lgbm.pkl")

    model = train_lgbm_model(df, FEATURE_COLS)
    save_model(model, model_path)
    loaded = load_model(model_path)

    X = df[FEATURE_COLS].values
    probs_original = model.predict_proba(X)[:, 1]
    probs_loaded = loaded.predict_proba(X)[:, 1]

    np.testing.assert_array_almost_equal(probs_original, probs_loaded, decimal=5)
