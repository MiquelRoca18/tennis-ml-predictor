"""
Módulo de Demos - Kelly Criterion
=================================

Demo refactorizada de Kelly Criterion.
"""

from src.utils import print_header, print_metric
from src.betting import KellyCalculator, BankrollSimulator
import pandas as pd
import numpy as np


def demo_kelly():
    """Demo de Kelly Criterion"""
    print_header("DEMO - KELLY CRITERION", "💎")
    
    try:
        # Generar datos de ejemplo
        print("\n📊 Generando datos de ejemplo...")
        np.random.seed(42)
        
        df = pd.DataFrame({
            'prob_modelo': np.random.uniform(0.52, 0.68, 50),
            'cuota': np.random.uniform(1.80, 2.30, 50),
            'resultado': np.random.binomial(1, 0.55, 50)
        })
        
        df = df[df['prob_modelo'] * df['cuota'] > 1.0].copy()
        
        print(f"✅ Generadas {len(df)} apuestas con valor positivo")
        
        # Simular con Kelly
        print_header("SIMULACIÓN CON KELLY", "🎲")
        
        sim = BankrollSimulator(bankroll_inicial=1000)
        resultados = sim.comparar_estrategias(df, apuesta_flat=10)
        
        # Mostrar resultados
        print_header("RESULTADOS", "📊")
        print_metric("Flat Betting ROI", resultados['flat']['roi'], "%", 2)
        print_metric("Kelly Criterion ROI", resultados['kelly']['roi'], "%", 2)
        
        mejora = resultados['kelly']['roi'] - resultados['flat']['roi']
        print_metric("Mejora con Kelly", mejora, "%", 2)
        
        print("\n✅ Demo de Kelly completada!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
