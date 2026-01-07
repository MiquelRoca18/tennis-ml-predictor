"""
Script simplificado de backtesting para el pipeline de setup
Ejecuta backtesting básico con el modelo entrenado
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import sys

# Añadir path para imports
sys.path.insert(0, str(Path(__file__).parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ejecutar_backtesting_simple():
    """
    Backtesting simple usando el dataset de features ya generado
    """
    logger.info("="*70)
    logger.info("🎲 BACKTESTING SIMPLE - 2024")
    logger.info("="*70)
    
    # Cargar modelo
    logger.info("\n📂 Cargando modelo...")
    modelo_path = Path("modelos/random_forest_calibrado.pkl")
    
    if not modelo_path.exists():
        logger.error(f"❌ Modelo no encontrado: {modelo_path}")
        logger.info("💡 Ejecuta primero el entrenamiento del modelo")
        return False
    
    modelo = joblib.load(modelo_path)
    logger.info("✅ Modelo cargado")
    
    # Cargar features seleccionadas
    logger.info("\n📂 Cargando features...")
    with open('resultados/selected_features.txt', 'r') as f:
        feature_cols = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"✅ {len(feature_cols)} features cargadas")
    
    # Cargar dataset
    logger.info("\n📂 Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    logger.info(f"✅ {len(df)} partidos cargados")
    
    # Filtrar solo 2024 para backtesting
    df['fecha'] = pd.to_datetime(df['fecha'])
    df_2024 = df[df['fecha'].dt.year == 2024].copy()
    logger.info(f"✅ {len(df_2024)} partidos de 2024")
    
    if len(df_2024) == 0:
        logger.warning("⚠️  No hay partidos de 2024 en el dataset")
        logger.info("✅ Backtesting completado (sin datos de 2024)")
        return True
    
    # Preparar datos
    X = df_2024[feature_cols]
    y = df_2024['ganador_j1']
    
    # Hacer predicciones
    logger.info("\n🔮 Generando predicciones...")
    y_prob = modelo.predict_proba(X)[:, 1]
    y_pred = modelo.predict(X)
    
    # Calcular métricas
    accuracy = (y_pred == y).mean()
    
    # Simular apuestas simples (flat betting)
    logger.info("\n💰 Simulando apuestas...")
    bankroll_inicial = 1000.0
    stake_por_apuesta = 10.0
    umbral_prob = 0.60  # Solo apostar cuando modelo está seguro
    
    # Filtrar solo apuestas con alta confianza
    apuestas_mask = (y_prob > umbral_prob) | (y_prob < (1 - umbral_prob))
    
    n_apuestas = apuestas_mask.sum()
    if n_apuestas == 0:
        logger.warning("⚠️  No se encontraron apuestas con suficiente confianza")
        logger.info("✅ Backtesting completado")
        return True
    
    # Calcular resultados
    aciertos = ((y_pred == y) & apuestas_mask).sum()
    win_rate = aciertos / n_apuestas if n_apuestas > 0 else 0
    
    # Simular ganancia (asumiendo cuota promedio de 1.8)
    cuota_promedio = 1.8
    ganancia_por_acierto = stake_por_apuesta * (cuota_promedio - 1)
    perdida_por_fallo = -stake_por_apuesta
    
    ganancia_total = (aciertos * ganancia_por_acierto) + ((n_apuestas - aciertos) * perdida_por_fallo)
    roi = (ganancia_total / (n_apuestas * stake_por_apuesta)) * 100
    
    # Mostrar resultados
    logger.info("\n" + "="*70)
    logger.info("📊 RESULTADOS DEL BACKTESTING")
    logger.info("="*70)
    logger.info(f"\n🎯 ACCURACY:")
    logger.info(f"  Accuracy general: {accuracy*100:.2f}%")
    logger.info(f"\n💰 SIMULACIÓN DE APUESTAS:")
    logger.info(f"  Bankroll inicial: {bankroll_inicial:.2f}€")
    logger.info(f"  Apuestas realizadas: {n_apuestas}")
    logger.info(f"  Apuestas ganadas: {aciertos} ({win_rate*100:.1f}%)")
    logger.info(f"  Apuestas perdidas: {n_apuestas - aciertos}")
    logger.info(f"  Ganancia total: {ganancia_total:+.2f}€")
    logger.info(f"  ROI: {roi:+.2f}%")
    logger.info(f"  Bankroll final: {bankroll_inicial + ganancia_total:.2f}€")
    logger.info("="*70)
    
    # Guardar resultados
    resultados_dir = Path("resultados/backtesting")
    resultados_dir.mkdir(parents=True, exist_ok=True)
    
    resultados = {
        'accuracy': accuracy,
        'n_apuestas': n_apuestas,
        'win_rate': win_rate,
        'roi': roi,
        'ganancia_total': ganancia_total
    }
    
    pd.DataFrame([resultados]).to_csv(
        resultados_dir / "backtesting_2024_simple.csv",
        index=False
    )
    
    logger.info(f"\n💾 Resultados guardados en: {resultados_dir}")
    logger.info("\n✅ Backtesting completado exitosamente")
    
    return True


if __name__ == "__main__":
    try:
        success = ejecutar_backtesting_simple()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Error en backtesting: {e}", exc_info=True)
        sys.exit(1)
