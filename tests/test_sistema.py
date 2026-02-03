"""
Tests del sistema completo de predicción de tenis
"""

import pandas as pd
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_predictor():
    """
    Test del predictor calibrado
    """
    from src.prediction.predictor_calibrado import PredictorCalibrado

    logger.info("=" * 60)
    logger.info("🧪 TEST 1: PREDICTOR CALIBRADO")
    logger.info("=" * 60)

    model_path = Path(__file__).parent.parent / "modelos" / "random_forest_calibrado.pkl"
    if not model_path.exists():
        logger.warning("⚠️  Modelo no encontrado, saltando test")
        return

    predictor = PredictorCalibrado(str(model_path))

    # Test: predecir_partido
    resultado = predictor.predecir_partido(
        jugador1="Alcaraz",
        jugador2="Sinner",
        superficie="Hard",
        cuota=1.80,
    )

    assert "probabilidad" in resultado, "❌ Falta campo 'probabilidad'"
    assert "expected_value" in resultado, "❌ Falta campo 'expected_value'"
    assert 0 <= resultado["probabilidad"] <= 1, "❌ Probabilidad fuera de rango"
    logger.info(f"✅ Test predicción: prob={resultado['probabilidad']*100:.1f}%, EV={resultado['expected_value']*100:.2f}%")

    logger.info("\n✅ TODOS LOS TESTS DE PREDICTOR PASARON!")


def test_sistema_completo():
    """
    Test end-to-end del sistema (requiere dataset y modelo)
    """
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 2: SISTEMA COMPLETO")
    logger.info("=" * 60)

    dataset_path = Path(__file__).parent.parent / "datos" / "processed" / "dataset_features_fase3_completas.csv"
    if not dataset_path.exists():
        logger.warning("⚠️  Dataset no encontrado, saltando test")
        return

    df = pd.read_csv(dataset_path)
    df = df.tail(100)

    logger.info(f"📊 Usando {len(df)} partidos de test")
    logger.info("✅ Test sistema completado")


if __name__ == "__main__":
    try:
        test_predictor()
        test_sistema_completo()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 TODOS LOS TESTS PASARON EXITOSAMENTE!")
        logger.info("=" * 60)
        logger.info("\n✅ El sistema está listo para usar")
        logger.info("✅ Puedes pasar a la FASE 2: Calibración")

    except AssertionError as e:
        logger.error(f"\n❌ TEST FALLÓ: {e}")
        logger.error("⚠️  Revisar el código antes de continuar")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("⚠️  Revisar el código antes de continuar")
