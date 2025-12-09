"""
Motor de decisión para apuestas en tenis
"""
import pandas as pd
import sys
from pathlib import Path

# Añadir el directorio src al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from models.predictor import TennisPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingDecisionEngine:
    """
    Motor de decisión para apuestas en tenis
    """
    
    def __init__(self, modelo_path="modelos/modelo_rf_v1.pkl", umbral_ev=0.03):
        """
        Args:
            modelo_path: ruta al modelo
            umbral_ev: EV mínimo para apostar (default: 3%)
        """
        self.predictor = TennisPredictor(modelo_path)
        self.umbral_ev = umbral_ev
    
    def evaluar_jornada(self, partidos_df):
        """
        Evalúa una jornada completa de partidos
        
        Args:
            partidos_df: DataFrame con columnas:
                - jugador_nombre
                - jugador_rank
                - oponente_nombre
                - oponente_rank
                - superficie
                - cuota_jugador
        
        Returns:
            DataFrame con análisis de todos los partidos
        """
        
        resultados = []
        
        for idx, partido in partidos_df.iterrows():
            resultado = self.predictor.analizar_partido(
                jugador_nombre=partido['jugador_nombre'],
                jugador_rank=partido['jugador_rank'],
                oponente_nombre=partido['oponente_nombre'],
                oponente_rank=partido['oponente_rank'],
                superficie=partido['superficie'],
                cuota_jugador=partido['cuota_jugador'],
                umbral_ev=self.umbral_ev
            )
            resultados.append(resultado)
        
        return pd.DataFrame(resultados)
    
    def filtrar_apuestas(self, resultados_df):
        """
        Filtra solo las apuestas que superan el umbral de EV
        
        Args:
            resultados_df: DataFrame con resultados de evaluación
            
        Returns:
            DataFrame con solo apuestas recomendadas
        """
        apuestas = resultados_df[resultados_df['decision'].str.contains('APOSTAR')]
        return apuestas
    
    def mostrar_resumen_jornada(self, resultados_df):
        """
        Muestra resumen de la jornada
        
        Args:
            resultados_df: DataFrame con resultados
        """
        
        apuestas = self.filtrar_apuestas(resultados_df)
        
        print("\n" + "=" * 70)
        print("📅 RESUMEN DE LA JORNADA")
        print("=" * 70)
        print(f"\n📊 Total de partidos evaluados: {len(resultados_df)}")
        print(f"✅ Apuestas recomendadas: {len(apuestas)} ({len(apuestas)/len(resultados_df)*100:.1f}%)")
        print(f"❌ Sin valor: {len(resultados_df) - len(apuestas)}")
        
        if len(apuestas) > 0:
            print(f"\n💰 EV promedio de apuestas: {apuestas['ev_pct'].mean():.2f}%")
            print(f"📈 Mejor oportunidad: {apuestas['ev_pct'].max():.2f}%")
            
            print("\n🎯 APUESTAS RECOMENDADAS:")
            print("-" * 70)
            for idx, apuesta in apuestas.iterrows():
                print(f"\n{idx+1}. {apuesta['jugador']} vs {apuesta['oponente']}")
                print(f"   Superficie: {apuesta['superficie']} | Cuota: @{apuesta['cuota']:.2f}")
                print(f"   Probabilidad: {apuesta['prob_modelo']*100:.1f}% | EV: +{apuesta['ev_pct']:.2f}%")
        else:
            print("\n⚠️  No hay apuestas con valor suficiente en esta jornada")
        
        print("=" * 70)


if __name__ == "__main__":
    # Crear motor de decisión
    engine = BettingDecisionEngine(
        modelo_path="modelos/modelo_rf_v1.pkl",
        umbral_ev=0.03  # 3% mínimo
    )
    
    # Jornada de ejemplo
    partidos_hoy = pd.DataFrame([
        {
            'jugador_nombre': 'Carlos Alcaraz',
            'jugador_rank': 3,
            'oponente_nombre': 'Jannik Sinner',
            'oponente_rank': 1,
            'superficie': 'Hard',
            'cuota_jugador': 2.10
        },
        {
            'jugador_nombre': 'Novak Djokovic',
            'jugador_rank': 7,
            'oponente_nombre': 'Daniil Medvedev',
            'oponente_rank': 5,
            'superficie': 'Hard',
            'cuota_jugador': 1.75
        },
        {
            'jugador_nombre': 'Lorenzo Musetti',
            'jugador_rank': 17,
            'oponente_nombre': 'Stefanos Tsitsipas',
            'oponente_rank': 11,
            'superficie': 'Clay',
            'cuota_jugador': 3.20
        },
        {
            'jugador_nombre': 'Alexander Zverev',
            'jugador_rank': 2,
            'oponente_nombre': 'Grigor Dimitrov',
            'oponente_rank': 10,
            'superficie': 'Grass',
            'cuota_jugador': 1.40
        },
        {
            'jugador_nombre': 'Taylor Fritz',
            'jugador_rank': 4,
            'oponente_nombre': 'Hubert Hurkacz',
            'oponente_rank': 16,
            'superficie': 'Hard',
            'cuota_jugador': 1.65
        }
    ])
    
    print("🎾 Evaluando jornada de partidos...")
    
    # Evaluar todos los partidos
    resultados = engine.evaluar_jornada(partidos_hoy)
    
    # Mostrar resumen
    engine.mostrar_resumen_jornada(resultados)
    
    # Guardar resultados
    output_dir = Path("resultados")
    output_dir.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(output_dir / 'analisis_jornada.csv', index=False)
    print(f"\n💾 Resultados guardados en: {output_dir / 'analisis_jornada.csv'}")
