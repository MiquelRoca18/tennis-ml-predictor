"""
Análisis por Categorías
Fase 4: Tennis ML Predictor
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Añadir path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.database_setup import TennisDatabase


class AnalisisCategorias:
    """
    Análisis detallado por diferentes categorías
    """
    
    def __init__(self, db_path="apuestas_tracker.db"):
        self.db = TennisDatabase(db_path)
    
    def analisis_por_superficie(self):
        """Análisis detallado por superficie"""
        
        df = self.db.obtener_predicciones()
        df = df[(df['decision'] == 'APOSTAR') & (df['resultado_real'].notna())]
        
        if len(df) == 0:
            print("⚠️  No hay datos suficientes")
            return
        
        print("=" * 60)
        print("🎾 ANÁLISIS POR SUPERFICIE")
        print("=" * 60)
        
        for superficie in ['Hard', 'Clay', 'Grass']:
            df_sup = df[df['superficie'] == superficie]
            
            if len(df_sup) == 0:
                continue
            
            total = len(df_sup)
            ganadas = (df_sup['resultado_real'] == 1).sum()
            win_rate = ganadas / total * 100
            
            total_apostado = df_sup['apuesta_cantidad'].sum()
            ganancia_neta = df_sup['ganancia'].sum()
            roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0
            
            print(f"\n{superficie}:")
            print(f"   Apuestas: {total}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   ROI: {roi:+.1f}%")
            print(f"   Ganancia: {ganancia_neta:+.2f}€")
    
    def analisis_por_ranking(self):
        """Análisis por categorías de ranking"""
        
        df = self.db.obtener_predicciones()
        df = df[(df['decision'] == 'APOSTAR') & (df['resultado_real'].notna())]
        
        if len(df) == 0:
            print("⚠️  No hay datos suficientes")
            return
        
        print("\n" + "=" * 60)
        print("🏆 ANÁLISIS POR CATEGORÍA DE RANKING")
        print("=" * 60)
        
        # Definir categorías
        df['categoria_ranking'] = pd.cut(
            df['jugador_rank'],
            bins=[0, 10, 50, 100, 500],
            labels=['Top 10', 'Top 11-50', 'Top 51-100', 'Top 100+']
        )
        
        for categoria in ['Top 10', 'Top 11-50', 'Top 51-100', 'Top 100+']:
            df_cat = df[df['categoria_ranking'] == categoria]
            
            if len(df_cat) == 0:
                continue
            
            total = len(df_cat)
            ganadas = (df_cat['resultado_real'] == 1).sum()
            win_rate = ganadas / total * 100
            
            total_apostado = df_cat['apuesta_cantidad'].sum()
            ganancia_neta = df_cat['ganancia'].sum()
            roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0
            
            print(f"\n{categoria}:")
            print(f"   Apuestas: {total}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   ROI: {roi:+.1f}%")
    
    def analisis_por_ev_range(self):
        """Análisis por rango de EV"""
        
        df = self.db.obtener_predicciones()
        df = df[(df['decision'] == 'APOSTAR') & (df['resultado_real'].notna())]
        
        if len(df) == 0:
            print("⚠️  No hay datos suficientes")
            return
        
        print("\n" + "=" * 60)
        print("📈 ANÁLISIS POR RANGO DE EV")
        print("=" * 60)
        
        # Definir rangos
        df['ev_range'] = pd.cut(
            df['ev'] * 100,
            bins=[0, 3, 5, 10, 100],
            labels=['0-3%', '3-5%', '5-10%', '>10%']
        )
        
        for rango in ['0-3%', '3-5%', '5-10%', '>10%']:
            df_rango = df[df['ev_range'] == rango]
            
            if len(df_rango) == 0:
                continue
            
            total = len(df_rango)
            ganadas = (df_rango['resultado_real'] == 1).sum()
            win_rate = ganadas / total * 100
            
            total_apostado = df_rango['apuesta_cantidad'].sum()
            ganancia_neta = df_rango['ganancia'].sum()
            roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0
            
            ev_promedio = df_rango['ev'].mean() * 100
            
            print(f"\nEV {rango}:")
            print(f"   Apuestas: {total}")
            print(f"   EV promedio: {ev_promedio:.2f}%")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   ROI: {roi:+.1f}%")
    
    def analisis_por_cuota(self):
        """Análisis por rango de cuotas"""
        
        df = self.db.obtener_predicciones()
        df = df[(df['decision'] == 'APOSTAR') & (df['resultado_real'].notna())]
        
        if len(df) == 0:
            print("⚠️  No hay datos suficientes")
            return
        
        print("\n" + "=" * 60)
        print("💰 ANÁLISIS POR RANGO DE CUOTAS")
        print("=" * 60)
        
        # Definir rangos
        df['cuota_range'] = pd.cut(
            df['cuota'],
            bins=[0, 1.5, 2.0, 3.0, 10.0],
            labels=['<1.5 (Favoritos)', '1.5-2.0', '2.0-3.0', '>3.0 (Underdogs)']
        )
        
        for rango in ['<1.5 (Favoritos)', '1.5-2.0', '2.0-3.0', '>3.0 (Underdogs)']:
            df_rango = df[df['cuota_range'] == rango]
            
            if len(df_rango) == 0:
                continue
            
            total = len(df_rango)
            ganadas = (df_rango['resultado_real'] == 1).sum()
            win_rate = ganadas / total * 100
            
            total_apostado = df_rango['apuesta_cantidad'].sum()
            ganancia_neta = df_rango['ganancia'].sum()
            roi = (ganancia_neta / total_apostado) * 100 if total_apostado > 0 else 0
            
            cuota_promedio = df_rango['cuota'].mean()
            
            print(f"\nCuotas {rango}:")
            print(f"   Apuestas: {total}")
            print(f"   Cuota promedio: {cuota_promedio:.2f}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   ROI: {roi:+.1f}%")
    
    def generar_reporte_completo(self):
        """Genera reporte completo de todas las categorías"""
        
        print("\n" + "=" * 60)
        print("📊 REPORTE COMPLETO DE ANÁLISIS POR CATEGORÍAS")
        print("=" * 60)
        
        self.analisis_por_superficie()
        self.analisis_por_ranking()
        self.analisis_por_ev_range()
        self.analisis_por_cuota()
        
        print("\n" + "=" * 60)
        print("✅ Análisis completo finalizado")
        print("=" * 60)


# Ejecutar
if __name__ == "__main__":
    analisis = AnalisisCategorias("apuestas_tracker.db")
    analisis.generar_reporte_completo()
