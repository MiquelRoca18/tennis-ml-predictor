"""
Sistema de Tracking Integrado con Predictor
Fase 4: Tennis ML Predictor
"""

import pandas as pd
from pathlib import Path
import sys

# Añadir path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tracking.database_setup import TennisDatabase
from predictor_calibrado import PredictorCalibrado
from datetime import datetime


class TrackingSystem:
    """
    Sistema integrado de predicción y tracking
    """
    
    def __init__(self, modelo_path, db_path="apuestas_tracker.db"):
        """
        Inicializa el sistema
        
        Args:
            modelo_path: Path al modelo calibrado
            db_path: Path a la base de datos SQLite
        """
        self.predictor = PredictorCalibrado(modelo_path)
        self.db = TennisDatabase(db_path)
        self.modelo_usado = Path(modelo_path).stem
    
    def predecir_y_registrar(self, partido_info, umbral_ev=0.03):
        """
        Predice un partido y registra en la base de datos
        
        Args:
            partido_info: dict con información del partido
                {
                    'fecha_partido': '2024-12-10',
                    'jugador_nombre': 'Alcaraz',
                    'jugador_rank': 3,
                    'oponente_nombre': 'Sinner',
                    'oponente_rank': 1,
                    'superficie': 'Hard',
                    'torneo': 'ATP Finals',
                    'cuota': 2.10,
                    'bookmaker': 'Bet365',
                    'features': {...}  # Features preparadas
                }
            umbral_ev: EV mínimo para apostar
        
        Returns:
            dict con resultado completo
        """
        
        # Realizar predicción
        resultado = self.predictor.recomendar_apuesta(
            features=partido_info['features'],
            cuota=partido_info['cuota'],
            umbral_ev=umbral_ev
        )
        
        # Preparar para insertar en DB
        db_entry = {
            'fecha_partido': partido_info['fecha_partido'],
            'jugador_nombre': partido_info['jugador_nombre'],
            'jugador_rank': partido_info.get('jugador_rank'),
            'oponente_nombre': partido_info['oponente_nombre'],
            'oponente_rank': partido_info.get('oponente_rank'),
            'superficie': partido_info.get('superficie'),
            'torneo': partido_info.get('torneo'),
            'ronda': partido_info.get('ronda'),
            'prob_modelo': resultado['probabilidad'],
            'prob_modelo_calibrada': resultado['probabilidad'],  # Ya está calibrada
            'cuota': resultado['cuota'],
            'bookmaker': partido_info.get('bookmaker'),
            'ev': resultado['ev'],
            'umbral_ev': umbral_ev,
            'decision': resultado['decision'],
            'modelo_usado': self.modelo_usado,
            'version_modelo': 'v3.0'
        }
        
        # Si decidimos apostar, añadir cantidad
        if resultado['apostar']:
            db_entry['apuesta_cantidad'] = resultado.get('stake', 10.0)
        
        # Insertar en DB
        prediccion_id = self.db.insertar_prediccion(db_entry)
        
        # Añadir ID al resultado
        resultado['prediccion_id'] = prediccion_id
        
        return resultado
    
    def procesar_jornada(self, partidos_df, umbral_ev=0.03):
        """
        Procesa una jornada completa de partidos
        
        Args:
            partidos_df: DataFrame con partidos del día
                Columnas: fecha_partido, jugador_nombre, jugador_rank,
                         oponente_nombre, oponente_rank, superficie, cuota, features, etc.
        
        Returns:
            DataFrame con resultados
        """
        
        print("=" * 60)
        print("📅 PROCESANDO JORNADA")
        print("=" * 60)
        print(f"\n📊 Total de partidos: {len(partidos_df)}")
        
        resultados = []
        
        for idx, row in partidos_df.iterrows():
            print(f"\n{idx+1}. {row['jugador_nombre']} vs {row['oponente_nombre']}")
            
            resultado = self.predecir_y_registrar(
                partido_info=row.to_dict(),
                umbral_ev=umbral_ev
            )
            
            resultados.append(resultado)
            
            # Mostrar resumen
            if resultado['apostar']:
                print(f"   ✅ APOSTAR - EV: +{resultado['ev']*100:.2f}%")
            else:
                print(f"   ❌ NO APOSTAR - EV: {resultado['ev']*100:+.2f}%")
        
        df_resultados = pd.DataFrame(resultados)
        
        # Resumen
        apuestas = df_resultados[df_resultados['apostar'] == True]
        
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE LA JORNADA")
        print("=" * 60)
        print(f"Total evaluado: {len(df_resultados)}")
        print(f"Apuestas recomendadas: {len(apuestas)} ({len(apuestas)/len(df_resultados)*100:.1f}%)")
        
        if len(apuestas) > 0:
            print(f"EV promedio: +{apuestas['ev'].mean()*100:.2f}%")
            print(f"Total a apostar: {len(apuestas) * 10:.0f}€")
        
        return df_resultados
    
    def actualizar_resultados_batch(self, resultados_reales):
        """
        Actualiza resultados de múltiples partidos
        
        Args:
            resultados_reales: DataFrame con columnas:
                - prediccion_id
                - resultado (1 o 0)
        """
        
        print("=" * 60)
        print("🔄 ACTUALIZANDO RESULTADOS")
        print("=" * 60)
        
        for idx, row in resultados_reales.iterrows():
            # Obtener info de la predicción
            df_pred = self.db.obtener_predicciones()
            df_pred = df_pred[df_pred['id'] == row['prediccion_id']]
            
            if len(df_pred) == 0:
                print(f"⚠️  Predicción {row['prediccion_id']} no encontrada")
                continue
            
            pred = df_pred.iloc[0]
            
            # Calcular ganancia si hubo apuesta
            if pred['decision'] == 'APOSTAR' or pred['apuesta_cantidad'] is not None:
                if row['resultado'] == 1:
                    ganancia = pred['apuesta_cantidad'] * (pred['cuota'] - 1)
                else:
                    ganancia = -pred['apuesta_cantidad']
            else:
                ganancia = None
            
            # Actualizar
            self.db.actualizar_resultado(
                prediccion_id=row['prediccion_id'],
                resultado=row['resultado'],
                ganancia=ganancia
            )
            
            resultado_str = "✅ GANÓ" if row['resultado'] == 1 else "❌ PERDIÓ"
            print(f"ID {row['prediccion_id']}: {resultado_str}, Ganancia: {ganancia}€")
        
        print(f"\n✅ {len(resultados_reales)} resultados actualizados")
    
    def generar_reporte(self):
        """
        Genera un reporte completo del sistema
        """
        
        metricas = self.db.calcular_metricas()
        
        print("\n" + "=" * 60)
        print("📊 REPORTE COMPLETO")
        print("=" * 60)
        
        print(f"\n💰 FINANCIERO:")
        print(f"   Total apostado:   {metricas['total_apostado']:.2f}€")
        print(f"   Total retornado:  {metricas['total_ganado']:.2f}€")
        print(f"   Ganancia neta:    {metricas['ganancia_neta']:+.2f}€")
        print(f"   ROI:              {metricas['roi']:+.2f}%")
        
        print(f"\n🎯 PERFORMANCE:")
        print(f"   Apuestas totales: {metricas['total_apuestas']}")
        print(f"   Ganadas:          {metricas['apuestas_ganadas']}")
        print(f"   Perdidas:         {metricas['apuestas_perdidas']}")
        print(f"   Win Rate:         {metricas['win_rate']:.1f}%")
        
        print(f"\n📈 MODELO:")
        print(f"   EV promedio:      +{metricas['ev_promedio']:.2f}%")
        
        return metricas


# Ejemplo de uso
if __name__ == "__main__":
    # Crear sistema
    sistema = TrackingSystem(
        modelo_path="modelos/xgboost_optimizado_2022_2025.pkl",
        db_path="apuestas_tracker.db"
    )
    
    print("✅ Sistema de tracking inicializado")
    print(f"📁 Base de datos: apuestas_tracker.db")
    print(f"🤖 Modelo: xgboost_optimizado_2022_2025.pkl")
    
    # Generar reporte
    sistema.generar_reporte()
