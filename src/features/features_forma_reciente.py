"""
Calculador de forma reciente
Analiza rendimiento reciente de jugadores
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormaRecienteCalculator:
    """
    Calcula features de forma reciente para cada jugador
    """
    
    def __init__(self, df_partidos):
        """
        Args:
            df_partidos: DataFrame con partidos
                Columnas requeridas: tourney_date, winner_name, loser_name, score
        """
        self.df = df_partidos.copy()
        self.df['tourney_date'] = pd.to_datetime(self.df['tourney_date'])
    
    def calcular_forma(self, jugador_nombre, fecha_partido, ventana_dias=60):
        """
        Calcula forma del jugador en los últimos N días
        
        Args:
            jugador_nombre: Nombre del jugador
            fecha_partido: Fecha del partido a predecir
            ventana_dias: Ventana de tiempo (default: 60 días)
        
        Returns:
            dict con métricas de forma
        """
        
        # Filtrar partidos del jugador ANTES de esta fecha
        fecha_inicio = fecha_partido - timedelta(days=ventana_dias)
        
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
        
        # Combinar todos los partidos
        partidos_ganados['resultado'] = 1
        partidos_perdidos['resultado'] = 0
        
        partidos_recientes = pd.concat([
            partidos_ganados[['tourney_date', 'resultado']],
            partidos_perdidos[['tourney_date', 'resultado']]
        ]).sort_values('tourney_date')
        
        if len(partidos_recientes) == 0:
            return self._forma_default()
        
        # Métricas básicas
        n_partidos = len(partidos_recientes)
        victorias = (partidos_recientes['resultado'] == 1).sum()
        derrotas = n_partidos - victorias
        win_rate = victorias / n_partidos
        
        # Racha actual (victorias/derrotas consecutivas)
        racha = self._calcular_racha(partidos_recientes)
        
        # Victorias recientes ponderadas (más peso a partidos recientes)
        victorias_ponderadas = self._victorias_ponderadas(partidos_recientes)
        
        return {
            'partidos_ultimos_60d': n_partidos,
            'victorias_ultimos_60d': int(victorias),
            'win_rate_60d': win_rate,
            'racha_actual': racha,
            'victorias_ponderadas_60d': victorias_ponderadas
        }
    
    def _calcular_racha(self, df_partidos):
        """
        Calcula racha de victorias (+) o derrotas (-)
        
        Args:
            df_partidos: DataFrame con partidos ordenados por fecha
            
        Returns:
            int: racha (positivo = victorias, negativo = derrotas)
        """
        if len(df_partidos) == 0:
            return 0
        
        racha = 0
        ultimo_resultado = df_partidos.iloc[-1]['resultado']
        
        # Contar hacia atrás hasta encontrar resultado diferente
        for idx in range(len(df_partidos) - 1, -1, -1):
            if df_partidos.iloc[idx]['resultado'] == ultimo_resultado:
                racha += 1 if ultimo_resultado == 1 else -1
            else:
                break
        
        return racha
    
    def _victorias_ponderadas(self, df_partidos):
        """
        Victorias con más peso a partidos recientes
        
        Args:
            df_partidos: DataFrame con partidos ordenados por fecha
            
        Returns:
            float: win rate ponderado
        """
        if len(df_partidos) == 0:
            return 0.5
        
        # Pesos lineales: más reciente = más peso
        pesos = np.linspace(0.5, 1.0, len(df_partidos))
        victorias_ponderadas = (df_partidos['resultado'].values * pesos).sum()
        peso_total = pesos.sum()
        
        return victorias_ponderadas / peso_total
    
    def _forma_default(self):
        """Valores por defecto cuando no hay datos"""
        return {
            'partidos_ultimos_60d': 0,
            'victorias_ultimos_60d': 0,
            'win_rate_60d': 0.5,
            'racha_actual': 0,
            'victorias_ponderadas_60d': 0.5
        }


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    logger.info("Cargando datos...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    calc = FormaRecienteCalculator(df)
    
    # Buscar jugador de ejemplo
    jugadores_unicos = pd.concat([df['winner_name'], df['loser_name']]).unique()
    jugador_ejemplo = None
    for nombre in ['Carlos Alcaraz', 'Novak Djokovic', 'Rafael Nadal']:
        if nombre in jugadores_unicos:
            jugador_ejemplo = nombre
            break
    
    if jugador_ejemplo is None:
        jugador_ejemplo = jugadores_unicos[0]
    
    fecha_ejemplo = df['tourney_date'].max() - pd.Timedelta(days=30)
    
    logger.info("=" * 60)
    logger.info(f"📊 FORMA RECIENTE - {jugador_ejemplo}")
    logger.info("=" * 60)
    
    forma = calc.calcular_forma(
        jugador_nombre=jugador_ejemplo,
        fecha_partido=fecha_ejemplo,
        ventana_dias=60
    )
    
    for key, value in forma.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.3f}")
        else:
            logger.info(f"   {key}: {value}")
    
    logger.info("\n✅ Proceso completado!")
