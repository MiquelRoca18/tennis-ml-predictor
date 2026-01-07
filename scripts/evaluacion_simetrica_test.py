"""
Evaluación Simétrica del Test Set
==================================

Este script evalúa el modelo de forma más realista:
- Para cada partido, hace predicción bidireccional
- No asume que sabemos quién ganó de antemano
- Calcula el accuracy REAL del modelo

Esto nos dirá si el 70% de accuracy es real o inflado.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluar_modelo_simetrico():
    """
    Evalúa el modelo de forma simétrica (como en producción real)
    """
    
    logger.info("="*70)
    logger.info("🔍 EVALUACIÓN SIMÉTRICA DEL TEST SET")
    logger.info("="*70)
    
    # 1. Cargar datos
    logger.info("\n📂 Cargando datos...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # 2. Cargar features seleccionadas
    with open('resultados/selected_features.txt', 'r') as f:
        feature_cols = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"✅ Features: {len(feature_cols)}")
    
    # 3. Split (mismo que en entrenamiento)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    df_test = df.iloc[val_end:].copy()
    logger.info(f"✅ Test set: {len(df_test)} filas")
    
    # 4. Cargar modelo
    logger.info("\n📂 Cargando modelo...")
    modelo = joblib.load("modelos/random_forest_calibrado.pkl")
    logger.info("✅ Modelo cargado")
    
    # 5. Agrupar por partido (cada partido tiene 2 filas)
    logger.info("\n🔄 Agrupando partidos...")
    
    # Identificar partidos únicos por fecha
    # Asumimos que cada partido aparece 2 veces consecutivas
    partidos = []
    i = 0
    while i < len(df_test) - 1:
        fila1 = df_test.iloc[i]
        fila2 = df_test.iloc[i + 1]
        
        # Verificar que son el mismo partido (misma fecha, jugadores invertidos)
        if fila1['fecha'] == fila2['fecha']:
            # fila1 tiene resultado=1 (ganador como jugador)
            # fila2 tiene resultado=0 (perdedor como jugador)
            if fila1['resultado'] == 1 and fila2['resultado'] == 0:
                partidos.append({
                    'fecha': fila1['fecha'],
                    'fila_ganador': i,
                    'fila_perdedor': i + 1
                })
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    logger.info(f"✅ {len(partidos)} partidos identificados")
    
    # 6. Evaluar cada partido de forma simétrica
    logger.info("\n🎯 Evaluando partidos...")
    
    predicciones_correctas = 0
    total_partidos = len(partidos)
    
    probabilidades_ganador = []
    probabilidades_perdedor = []
    
    for idx, partido in enumerate(partidos):
        # Obtener las dos filas del partido
        fila_ganador = df_test.iloc[partido['fila_ganador']]
        fila_perdedor = df_test.iloc[partido['fila_perdedor']]
        
        # Predicción A: ganador como 'jugador'
        X_A = fila_ganador[feature_cols].values.reshape(1, -1)
        prob_A = modelo.predict_proba(X_A)[0, 1]  # P(ganador gana)
        
        # Predicción B: perdedor como 'jugador'
        X_B = fila_perdedor[feature_cols].values.reshape(1, -1)
        prob_B = modelo.predict_proba(X_B)[0, 1]  # P(perdedor gana)
        
        # Probabilidad final de que el ganador gane (promedio simétrico)
        prob_ganador_gana = (prob_A + (1 - prob_B)) / 2
        
        probabilidades_ganador.append(prob_ganador_gana)
        probabilidades_perdedor.append(1 - prob_ganador_gana)
        
        # Predicción: si prob_ganador_gana > 0.5, predecimos que gana el ganador
        if prob_ganador_gana > 0.5:
            predicciones_correctas += 1
        
        # Log progreso
        if (idx + 1) % 500 == 0:
            logger.info(f"  Procesados: {idx + 1}/{total_partidos}")
    
    # 7. Calcular métricas
    logger.info("\n" + "="*70)
    logger.info("📊 RESULTADOS")
    logger.info("="*70)
    
    accuracy_real = predicciones_correctas / total_partidos * 100
    
    logger.info(f"\n🎯 ACCURACY REAL (Evaluación Simétrica):")
    logger.info(f"   Partidos totales: {total_partidos}")
    logger.info(f"   Predicciones correctas: {predicciones_correctas}")
    logger.info(f"   Accuracy: {accuracy_real:.2f}%")
    
    # Comparar con accuracy "inflado"
    logger.info(f"\n📊 COMPARACIÓN:")
    logger.info(f"   Accuracy Test Set (método actual): 70.04%")
    logger.info(f"   Accuracy Real (evaluación simétrica): {accuracy_real:.2f}%")
    logger.info(f"   Diferencia: {70.04 - accuracy_real:.2f}%")
    
    # Distribución de probabilidades
    logger.info(f"\n📈 DISTRIBUCIÓN DE PROBABILIDADES:")
    logger.info(f"   Media (ganador): {np.mean(probabilidades_ganador):.3f}")
    logger.info(f"   Media (perdedor): {np.mean(probabilidades_perdedor):.3f}")
    logger.info(f"   Std (ganador): {np.std(probabilidades_ganador):.3f}")
    
    # Calibración
    logger.info(f"\n🎯 CALIBRACIÓN:")
    logger.info(f"   Si el modelo está bien calibrado:")
    logger.info(f"   - Probabilidad media del ganador debería ser ~{accuracy_real/100:.3f}")
    logger.info(f"   - Probabilidad media real: {np.mean(probabilidades_ganador):.3f}")
    
    if np.mean(probabilidades_ganador) > accuracy_real/100 + 0.05:
        logger.info(f"   ⚠️  Modelo OVERCONFIDENT (sobreestima probabilidades)")
    elif np.mean(probabilidades_ganador) < accuracy_real/100 - 0.05:
        logger.info(f"   ⚠️  Modelo UNDERCONFIDENT (subestima probabilidades)")
    else:
        logger.info(f"   ✅ Modelo bien calibrado")
    
    # Guardar resultados
    resultados = pd.DataFrame({
        'partido_num': range(len(partidos)),
        'fecha': [p['fecha'] for p in partidos],
        'prob_ganador': probabilidades_ganador,
        'prob_perdedor': probabilidades_perdedor,
        'prediccion_correcta': [p > 0.5 for p in probabilidades_ganador]
    })
    
    Path("resultados").mkdir(exist_ok=True)
    resultados.to_csv('resultados/evaluacion_simetrica.csv', index=False)
    logger.info(f"\n💾 Resultados guardados: resultados/evaluacion_simetrica.csv")
    
    logger.info("\n" + "="*70)
    logger.info("✅ EVALUACIÓN COMPLETADA")
    logger.info("="*70)
    
    return accuracy_real, np.mean(probabilidades_ganador)


if __name__ == "__main__":
    accuracy_real, prob_media = evaluar_modelo_simetrico()
    
    print(f"\n🎯 CONCLUSIÓN:")
    print(f"   El accuracy REAL del modelo es: {accuracy_real:.2f}%")
    print(f"   (No el 70% que mostraba el test set)")
