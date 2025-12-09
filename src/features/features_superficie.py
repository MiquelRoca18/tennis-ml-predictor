"""
Calculador de especialización por superficie
Identifica especialistas de cada superficie
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SuperficieSpecializationCalculator:
    """
    Calcula métricas de especialización en cada superficie
    """
    
    def __init__(self, df_partidos):
        """
        Args:
            df_partidos: DataFrame con partidos
                Columnas requeridas: tourney_date, winner_name, loser_name, surface
        """
        self.df = df_partidos.copy()
        self.df['tourney_date'] = pd.to_datetime(self.df['tourney_date'])
    
    def calcular_especializacion(self, jugador_nombre, fecha_partido, ventana_dias=365):
        """
        Calcula win rate en cada superficie
        
        Args:
            jugador_nombre: Nombre del jugador
            fecha_partido: Fecha del partido a predecir
            ventana_dias: Ventana de tiempo (default: 365 días)
        
        Returns:
            dict con win rates por superficie
        """
        
        fecha_inicio = fecha_partido - pd.Timedelta(days=ventana_dias)
        
        # Partidos ganados
        partidos_ganados = self.df[
            (self.df['winner_name'] == jugador_nombre) &
            (self.df['tourney_date'] >= fecha_inicio) &
            (self.df['tourney_date'] < fecha_partido)
        ].copy()
        
        # Partidos perdidos
        partidos_perdidos = self.df[
            (self.df['loser_name'] == jugador_nombre) &
            (self.df['tourney_date'] >= fecha_inicio) &
            (self.df['tourney_date'] < fecha_partido)
        ].copy()
        
        if len(partidos_ganados) == 0 and len(partidos_perdidos) == 0:
            return self._especializacion_default()
        
        # Win rate por superficie
        win_rates = {}
        for superficie in ['Hard', 'Clay', 'Grass']:
            ganados_sup = len(partidos_ganados[partidos_ganados['surface'] == superficie])
            perdidos_sup = len(partidos_perdidos[partidos_perdidos['surface'] == superficie])
            total_sup = ganados_sup + perdidos_sup
            
            if total_sup > 0:
                win_rates[f'win_rate_{superficie.lower()}'] = ganados_sup / total_sup
                win_rates[f'partidos_{superficie.lower()}'] = total_sup
            else:
                win_rates[f'win_rate_{superficie.lower()}'] = 0.5
                win_rates[f'partidos_{superficie.lower()}'] = 0
        
        # Superficie favorita (mayor win rate con min 5 partidos)
        mejor_superficie = None
        mejor_win_rate = 0
        
        for sup in ['Hard', 'Clay', 'Grass']:
            if (win_rates[f'partidos_{sup.lower()}'] >= 5 and 
                win_rates[f'win_rate_{sup.lower()}'] > mejor_win_rate):
                mejor_win_rate = win_rates[f'win_rate_{sup.lower()}']
                mejor_superficie = sup
        
        win_rates['superficie_favorita'] = mejor_superficie or 'Hard'
        
        return win_rates
    
    def calcular_ventaja_superficie(self, jugador1_nombre, jugador2_nombre, fecha_partido, superficie):
        """
        Calcula ventaja relativa en la superficie del partido
        
        Args:
            jugador1_nombre: Nombre del jugador 1
            jugador2_nombre: Nombre del jugador 2
            fecha_partido: Fecha del partido
            superficie: Superficie del partido
            
        Returns:
            dict con ventajas de superficie
        """
        
        esp_j1 = self.calcular_especializacion(jugador1_nombre, fecha_partido)
        esp_j2 = self.calcular_especializacion(jugador2_nombre, fecha_partido)
        
        win_rate_j1 = esp_j1[f'win_rate_{superficie.lower()}']
        win_rate_j2 = esp_j2[f'win_rate_{superficie.lower()}']
        
        ventaja = win_rate_j1 - win_rate_j2
        
        return {
            'ventaja_superficie': ventaja,
            'win_rate_j1_superficie': win_rate_j1,
            'win_rate_j2_superficie': win_rate_j2,
            'es_superficie_favorita_j1': 1 if esp_j1['superficie_favorita'] == superficie else 0,
            'es_superficie_favorita_j2': 1 if esp_j2['superficie_favorita'] == superficie else 0
        }
    
    def _especializacion_default(self):
        """Valores por defecto"""
        return {
            'win_rate_hard': 0.5,
            'win_rate_clay': 0.5,
            'win_rate_grass': 0.5,
            'partidos_hard': 0,
            'partidos_clay': 0,
            'partidos_grass': 0,
            'superficie_favorita': 'Hard'
        }


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logger.info("Cargando datos...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    calc = SuperficieSpecializationCalculator(df)
    
    # Buscar jugador de ejemplo
    jugadores_unicos = pd.concat([df['winner_name'], df['loser_name']]).unique()
    jugador_ejemplo = None
    for nombre in ['Rafael Nadal', 'Carlos Alcaraz', 'Novak Djokovic']:
        if nombre in jugadores_unicos:
            jugador_ejemplo = nombre
            break
    
    if jugador_ejemplo is None:
        jugador_ejemplo = jugadores_unicos[0]
    
    fecha_ejemplo = df['tourney_date'].max() - pd.Timedelta(days=30)
    
    logger.info("=" * 60)
    logger.info(f"🎯 ESPECIALIZACIÓN POR SUPERFICIE - {jugador_ejemplo}")
    logger.info("=" * 60)
    
    esp = calc.calcular_especializacion(
        jugador_nombre=jugador_ejemplo,
        fecha_partido=fecha_ejemplo,
        ventana_dias=365
    )
    
    for key, value in esp.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.3f}")
        else:
            logger.info(f"   {key}: {value}")
    
    logger.info("\n✅ Proceso completado!")
