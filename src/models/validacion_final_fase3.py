"""
Validación final del modelo optimizado - Fase 3
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validacion_completa(modelo_path, X_test, y_test, nombre_modelo="Modelo"):
    """
    Validación completa del modelo optimizado
    """
    
    logger.info("=" * 60)
    logger.info(f"✅ VALIDACIÓN FINAL - {nombre_modelo.upper()}")
    logger.info("=" * 60)
    
    # Cargar modelo
    modelo = joblib.load(modelo_path)
    
    # Predecir
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    
    logger.info(f"\n📊 MÉTRICAS FINALES:")
    logger.info(f"   Accuracy:     {accuracy*100:.2f}%")
    logger.info(f"   Brier Score:  {brier:.4f}")
    
    # Classification report
    logger.info(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Perdedor', 'Ganador']))
    
    # Calibración
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado', linewidth=2)
    plt.plot(prob_pred, prob_true, 'o-', label=nombre_modelo, linewidth=2, markersize=8)
    plt.xlabel('Probabilidad Predicha', fontsize=12)
    plt.ylabel('Fracción Positiva', fontsize=12)
    plt.title(f'Calibration Plot - {nombre_modelo}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    Path("resultados").mkdir(exist_ok=True)
    filename = f'resultados/calibration_final_{nombre_modelo.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    logger.info(f"\n📊 Gráfico guardado: {filename}")
    plt.close()
    
    # Verificar criterios de éxito
    logger.info("\n" + "=" * 60)
    logger.info("🎯 VERIFICACIÓN DE CRITERIOS")
    logger.info("=" * 60)
    
    criterios = {
        'Accuracy > 62%': accuracy > 0.62,
        'Brier Score < 0.18': brier < 0.18,
        'Brier Score < 0.20': brier < 0.20
    }
    
    for criterio, cumple in criterios.items():
        simbolo = "✅" if cumple else "❌"
        logger.info(f"{simbolo} {criterio}")
    
    if all(criterios.values()):
        logger.info("\n🎉 ¡TODOS LOS CRITERIOS CUMPLIDOS!")
        logger.info("✅ Listo para pasar a FASE 4")
    else:
        logger.info("\n⚠️  Algunos criterios no se cumplen")
        logger.info("💡 Considera ajustar hiperparámetros o añadir más features")
    
    return accuracy, brier


def comparar_fases(modelo_fase2_path, modelo_fase3_path, X_test, y_test):
    """
    Compara modelo de Fase 2 vs Fase 3
    """
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPARACIÓN FASE 2 vs FASE 3")
    logger.info("=" * 60)
    
    # Verificar si existe modelo de Fase 2
    if not Path(modelo_fase2_path).exists():
        logger.warning(f"⚠️  Modelo de Fase 2 no encontrado: {modelo_fase2_path}")
        logger.info("   Saltando comparación...")
        return
    
    # Fase 2
    logger.info("\n📌 FASE 2 (Modelo Base Calibrado):")
    acc_f2, brier_f2 = validacion_completa(modelo_fase2_path, X_test, y_test, "Fase 2")
    
    # Fase 3
    logger.info("\n📌 FASE 3 (Modelo Optimizado):")
    acc_f3, brier_f3 = validacion_completa(modelo_fase3_path, X_test, y_test, "Fase 3")
    
    # Comparación
    logger.info("\n" + "=" * 60)
    logger.info("🏆 MEJORAS DE FASE 3 SOBRE FASE 2")
    logger.info("=" * 60)
    
    mejora_acc = (acc_f3 - acc_f2) * 100
    mejora_brier = ((brier_f2 - brier_f3) / brier_f2) * 100
    
    logger.info(f"\n📈 Accuracy:")
    logger.info(f"   Fase 2: {acc_f2*100:.2f}%")
    logger.info(f"   Fase 3: {acc_f3*100:.2f}%")
    logger.info(f"   Mejora: {mejora_acc:+.2f}%")
    
    logger.info(f"\n📉 Brier Score:")
    logger.info(f"   Fase 2: {brier_f2:.4f}")
    logger.info(f"   Fase 3: {brier_f3:.4f}")
    logger.info(f"   Mejora: {mejora_brier:+.1f}%")
    
    if acc_f3 > acc_f2 and brier_f3 < brier_f2:
        logger.info("\n🎉 ¡FASE 3 MEJORA EN AMBAS MÉTRICAS!")
    elif acc_f3 > acc_f2:
        logger.info("\n✅ Mejora en Accuracy")
    elif brier_f3 < brier_f2:
        logger.info("\n✅ Mejora en Brier Score (calibración)")
    else:
        logger.info("\n⚠️  Sin mejoras significativas")


if __name__ == "__main__":
    # Cargar datos de test
    logger.info("📂 Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Cargar features seleccionadas
    selected_features_path = 'resultados/selected_features.txt'
    if not Path(selected_features_path).exists():
        logger.error("❌ No se encontraron features seleccionadas")
        logger.info("💡 Ejecuta primero: python run_fase3_optimization.py")
        exit(1)
    
    selected_features = pd.read_csv(selected_features_path, header=None)[0].tolist()
    logger.info(f"📋 Features seleccionadas: {len(selected_features)}")
    
    # Test set (últimos 20%)
    n = len(df)
    test_start = int(n * 0.8)
    
    X_test = df.iloc[test_start:][selected_features]
    y_test = df.iloc[test_start:]['resultado']
    
    logger.info(f"\n📊 Test set: {len(X_test)} partidos")
    
    # Buscar todos los modelos (calibrados y optimizados)
    modelos_calibrados = list(Path("modelos").glob("*_calibrado.pkl"))
    modelos_optimizados = list(Path("modelos").glob("*_optimizado.pkl"))
    
    todos_modelos = modelos_calibrados + modelos_optimizados
    
    if len(todos_modelos) == 0:
        logger.warning("⚠️  No se encontraron modelos para validar")
        logger.info("💡 Ejecuta primero: python run_fase3_optimization.py")
        exit(1)
    
    logger.info(f"\n🔍 Modelos encontrados: {len(todos_modelos)}")
    
    # Validar cada modelo
    for modelo_path in todos_modelos:
        nombre_base = modelo_path.stem.replace('_optimizado', '').replace('_calibrado', '').replace('_', ' ').title()
        tipo = "Optimizado" if "optimizado" in modelo_path.stem else "Calibrado"
        nombre_completo = f"{nombre_base} ({tipo})"
        
        try:
            validacion_completa(
                modelo_path=str(modelo_path),
                X_test=X_test,
                y_test=y_test,
                nombre_modelo=nombre_completo
            )
        except Exception as e:
            logger.error(f"❌ Error validando {nombre_completo}: {e}")
    
    logger.info("\n✅ Validación completada!")
