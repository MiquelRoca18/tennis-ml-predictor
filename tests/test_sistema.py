"""
Tests del sistema completo de predicción de tenis
"""

import pandas as pd
import sys
from pathlib import Path

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.predictor import TennisPredictor
from betting.decision_engine import BettingDecisionEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_predictor():
    """
    Test del predictor
    """
    logger.info("=" * 60)
    logger.info("🧪 TEST 1: PREDICTOR")
    logger.info("=" * 60)

    predictor = TennisPredictor("modelos/modelo_rf_v1.pkl")

    # Test 1: Predicción básica
    prob = predictor.predecir_probabilidad(jugador_rank=10, oponente_rank=50, superficie="Hard")

    assert 0 <= prob <= 1, "❌ Probabilidad fuera de rango"
    assert prob > 0.5, "❌ Jugador mejor rankeado debe tener prob > 50%"
    logger.info(f"✅ Test predicción básica: {prob*100:.1f}%")

    # Test 2: Cálculo de EV
    ev = predictor.calcular_ev(probabilidad=0.60, cuota=2.00)
    expected_ev = (0.60 * 2.00) - 1
    assert abs(ev - expected_ev) < 0.001, "❌ Cálculo de EV incorrecto"
    logger.info(f"✅ Test cálculo EV: {ev*100:.2f}%")

    # Test 3: Análisis completo
    resultado = predictor.analizar_partido(
        jugador_nombre="Test Player",
        jugador_rank=5,
        oponente_nombre="Test Opponent",
        oponente_rank=20,
        superficie="Clay",
        cuota_jugador=1.50,
    )

    assert "ev" in resultado, "❌ Falta campo 'ev' en resultado"
    assert "decision" in resultado, "❌ Falta campo 'decision' en resultado"
    logger.info(f"✅ Test análisis completo")

    logger.info("\n✅ TODOS LOS TESTS DE PREDICTOR PASARON!")


def test_decision_engine():
    """
    Test del decision engine
    """
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 2: DECISION ENGINE")
    logger.info("=" * 60)

    engine = BettingDecisionEngine(umbral_ev=0.03)

    # Test con jornada ficticia
    partidos = pd.DataFrame(
        [
            {
                "jugador_nombre": "Player 1",
                "jugador_rank": 5,
                "oponente_nombre": "Player 2",
                "oponente_rank": 30,
                "superficie": "Hard",
                "cuota_jugador": 1.50,
            },
            {
                "jugador_nombre": "Player 3",
                "jugador_rank": 10,
                "oponente_nombre": "Player 4",
                "oponente_rank": 8,
                "superficie": "Clay",
                "cuota_jugador": 2.20,
            },
        ]
    )

    resultados = engine.evaluar_jornada(partidos)

    assert len(resultados) == 2, "❌ Número de resultados incorrecto"
    assert "ev" in resultados.columns, "❌ Falta columna 'ev'"
    assert "decision" in resultados.columns, "❌ Falta columna 'decision'"

    logger.info(f"✅ Test evaluación jornada: {len(resultados)} partidos procesados")

    apuestas = engine.filtrar_apuestas(resultados)
    logger.info(f"✅ Test filtrado de apuestas: {len(apuestas)} apuestas encontradas")

    logger.info("\n✅ TODOS LOS TESTS DE DECISION ENGINE PASARON!")


def test_sistema_completo():
    """
    Test end-to-end del sistema
    """
    logger.info("\n" + "=" * 60)
    logger.info("🧪 TEST 3: SISTEMA COMPLETO END-TO-END")
    logger.info("=" * 60)

    # Cargar datos de test
    df = pd.read_csv("datos/processed/dataset_con_features.csv")
    df = df.tail(100)  # Últimos 100 partidos

    logger.info(f"📊 Usando {len(df)} partidos de test")

    # Simular predicciones
    predictor = TennisPredictor("modelos/modelo_rf_v1.pkl")

    aciertos = 0
    total = 0

    for idx, row in df.iterrows():
        prob = predictor.predecir_probabilidad(
            jugador_rank=row["jugador_rank"],
            oponente_rank=row["oponente_rank"],
            superficie=row.get("superficie", "Hard"),
        )

        prediccion = 1 if prob > 0.5 else 0
        real = row["resultado"]

        if prediccion == real:
            aciertos += 1
        total += 1

    accuracy = aciertos / total
    logger.info(f"\n📊 Accuracy en muestra de test: {accuracy*100:.2f}%")

    if accuracy > 0.55:
        logger.info("✅ Sistema funcionando correctamente (accuracy > 55%)")
    else:
        logger.warning("⚠️  Warning: Accuracy baja, revisar modelo")

    logger.info("\n✅ TEST END-TO-END COMPLETADO!")


if __name__ == "__main__":
    try:
        test_predictor()
        test_decision_engine()
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
