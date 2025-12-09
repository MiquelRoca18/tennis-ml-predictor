"""
Calculador de estadísticas de servicio y resto
Extrae métricas de servicio y resto de la base de datos TML
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServicioRestoCalculator:
    """
    Calcula estadísticas de servicio y resto
    
    Por qué son importantes:
    - El tenis es 50% servicio, 50% resto
    - Un sacador potente vs buen restador = matchup crítico
    - Estos datos YA ESTÁN en el dataset TML
    """
    
    def __init__(self, df_partidos):
        """
        Args:
            df_partidos: DataFrame con partidos de TML
                Columnas requeridas: tourney_date, winner_name, loser_name,
                w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
                l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced
        """
        self.df = df_partidos.copy()
        self.df['tourney_date'] = pd.to_datetime(self.df['tourney_date'])
    
    def calcular_estadisticas_servicio(self, jugador_nombre, fecha_partido, ventana_dias=90):
        """
        Calcula estadísticas de servicio de últimos N días
        
        Métricas clave:
        - % Aces
        - % Dobles faltas
        - % 1er servicio dentro
        - % puntos ganados con 1er servicio
        - % puntos ganados con 2do servicio
        - % break points salvados
        
        Args:
            jugador_nombre: Nombre del jugador
            fecha_partido: Fecha del partido a predecir
            ventana_dias: Ventana de tiempo en días (default: 90)
            
        Returns:
            dict con estadísticas de servicio
        """
        
        fecha_inicio = fecha_partido - pd.Timedelta(days=ventana_dias)
        
        # Filtrar partidos donde este jugador GANÓ
        partidos_ganados = self.df[
            (self.df['winner_name'] == jugador_nombre) &
            (self.df['tourney_date'] >= fecha_inicio) &
            (self.df['tourney_date'] < fecha_partido)
        ].copy()
        
        # Filtrar partidos donde este jugador PERDIÓ
        partidos_perdidos = self.df[
            (self.df['loser_name'] == jugador_nombre) &
            (self.df['tourney_date'] >= fecha_inicio) &
            (self.df['tourney_date'] < fecha_partido)
        ].copy()
        
        if len(partidos_ganados) == 0 and len(partidos_perdidos) == 0:
            return self._servicio_default()
        
        # Agregar estadísticas (w_ para ganados, l_ para perdidos)
        total_puntos_servicio = (
            partidos_ganados['w_svpt'].fillna(0).sum() + 
            partidos_perdidos['l_svpt'].fillna(0).sum()
        )
        
        if total_puntos_servicio == 0:
            return self._servicio_default()
        
        # Aces
        total_aces = partidos_ganados['w_ace'].fillna(0).sum() + partidos_perdidos['l_ace'].fillna(0).sum()
        
        # Dobles faltas
        total_df = partidos_ganados['w_df'].fillna(0).sum() + partidos_perdidos['l_df'].fillna(0).sum()
        
        # 1er servicio
        total_1stIn = partidos_ganados['w_1stIn'].fillna(0).sum() + partidos_perdidos['l_1stIn'].fillna(0).sum()
        total_1stWon = partidos_ganados['w_1stWon'].fillna(0).sum() + partidos_perdidos['l_1stWon'].fillna(0).sum()
        
        # 2do servicio
        total_2ndWon = partidos_ganados['w_2ndWon'].fillna(0).sum() + partidos_perdidos['l_2ndWon'].fillna(0).sum()
        puntos_2do_servicio = total_puntos_servicio - total_1stIn
        
        # Break points
        total_bpFaced = partidos_ganados['w_bpFaced'].fillna(0).sum() + partidos_perdidos['l_bpFaced'].fillna(0).sum()
        total_bpSaved = partidos_ganados['w_bpSaved'].fillna(0).sum() + partidos_perdidos['l_bpSaved'].fillna(0).sum()
        
        stats = {
            # Partidos incluidos
            'n_partidos': len(partidos_ganados) + len(partidos_perdidos),
            
            # Aces
            'aces_total': int(total_aces),
            'aces_pct': total_aces / total_puntos_servicio if total_puntos_servicio > 0 else 0.08,
            'aces_per_match': total_aces / (len(partidos_ganados) + len(partidos_perdidos)),
            
            # Dobles faltas
            'df_total': int(total_df),
            'df_pct': total_df / total_puntos_servicio if total_puntos_servicio > 0 else 0.04,
            
            # 1er servicio
            'first_serve_in_pct': total_1stIn / total_puntos_servicio if total_puntos_servicio > 0 else 0.62,
            'first_serve_won_pct': total_1stWon / total_1stIn if total_1stIn > 0 else 0.68,
            
            # 2do servicio
            'second_serve_won_pct': total_2ndWon / puntos_2do_servicio if puntos_2do_servicio > 0 else 0.48,
            
            # Break points
            'bp_faced': int(total_bpFaced),
            'bp_saved': int(total_bpSaved),
            'bp_saved_pct': total_bpSaved / total_bpFaced if total_bpFaced > 0 else 0.60,
            
            # Puntos totales de servicio
            'total_svpt': int(total_puntos_servicio)
        }
        
        return stats
    
    def calcular_estadisticas_resto(self, jugador_nombre, fecha_partido, ventana_dias=90):
        """
        Calcula estadísticas de resto (como RESTADOR, no sacador)
        
        Args:
            jugador_nombre: Nombre del jugador
            fecha_partido: Fecha del partido a predecir
            ventana_dias: Ventana de tiempo en días (default: 90)
            
        Returns:
            dict con estadísticas de resto
        """
        
        fecha_inicio = fecha_partido - pd.Timedelta(days=ventana_dias)
        
        # Cuando GANÓ (stats del oponente cuando restaba)
        partidos_ganados = self.df[
            (self.df['winner_name'] == jugador_nombre) &
            (self.df['tourney_date'] >= fecha_inicio) &
            (self.df['tourney_date'] < fecha_partido)
        ].copy()
        
        # Cuando PERDIÓ
        partidos_perdidos = self.df[
            (self.df['loser_name'] == jugador_nombre) &
            (self.df['tourney_date'] >= fecha_inicio) &
            (self.df['tourney_date'] < fecha_partido)
        ].copy()
        
        if len(partidos_ganados) == 0 and len(partidos_perdidos) == 0:
            return self._resto_default()
        
        # Break points CONVERTIDOS (cuando rompe el servicio del rival)
        # Cuando gana: los BP que enfrentó el rival menos los que salvó
        # Cuando pierde: los BP que enfrentó el rival menos los que salvó
        bp_opportunities = (
            partidos_ganados['l_bpFaced'].fillna(0).sum() + 
            partidos_perdidos['w_bpFaced'].fillna(0).sum()
        )
        bp_converted = (
            (partidos_ganados['l_bpFaced'].fillna(0) - partidos_ganados['l_bpSaved'].fillna(0)).sum() +
            (partidos_perdidos['w_bpFaced'].fillna(0) - partidos_perdidos['w_bpSaved'].fillna(0)).sum()
        )
        
        stats = {
            'n_partidos': len(partidos_ganados) + len(partidos_perdidos),
            
            # Break points
            'bp_opportunities': int(bp_opportunities),
            'bp_converted': int(bp_converted),
            'bp_conversion_pct': bp_converted / bp_opportunities if bp_opportunities > 0 else 0.35,
            
            # Score de calidad de resto
            'return_quality_score': bp_converted / bp_opportunities if bp_opportunities > 0 else 0.35
        }
        
        return stats
    
    def calcular_matchup_servicio_resto(self, jugador1_nombre, jugador2_nombre, fecha_partido):
        """
        Calcula matchup: servicio J1 vs resto J2
        
        CRÍTICO: Sacador potente vs buen restador
        
        Args:
            jugador1_nombre: Nombre del jugador 1
            jugador2_nombre: Nombre del jugador 2
            fecha_partido: Fecha del partido
            
        Returns:
            dict con ventajas de matchup
        """
        
        serv_j1 = self.calcular_estadisticas_servicio(jugador1_nombre, fecha_partido)
        resto_j2 = self.calcular_estadisticas_resto(jugador2_nombre, fecha_partido)
        
        # Ventaja en servicio (simplified score)
        serve_power_j1 = (
            serv_j1['aces_pct'] * 0.3 + 
            serv_j1['first_serve_won_pct'] * 0.5 + 
            serv_j1['bp_saved_pct'] * 0.2
        )
        
        return_power_j2 = resto_j2['bp_conversion_pct']
        
        ventaja = {
            'j1_serve_power': serve_power_j1,
            'j2_return_power': return_power_j2,
            'serve_vs_return_advantage': serve_power_j1 - return_power_j2
        }
        
        return ventaja
    
    def _servicio_default(self):
        """Valores por defecto (tour average)"""
        return {
            'n_partidos': 0,
            'aces_pct': 0.08,
            'aces_per_match': 5.0,
            'df_pct': 0.04,
            'first_serve_in_pct': 0.62,
            'first_serve_won_pct': 0.68,
            'second_serve_won_pct': 0.48,
            'bp_saved_pct': 0.60,
            'aces_total': 0,
            'df_total': 0,
            'bp_faced': 0,
            'bp_saved': 0,
            'total_svpt': 0
        }
    
    def _resto_default(self):
        """Valores por defecto"""
        return {
            'n_partidos': 0,
            'bp_opportunities': 0,
            'bp_converted': 0,
            'bp_conversion_pct': 0.35,
            'return_quality_score': 0.35
        }


# Ejemplo de uso
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Añadir path para imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Cargar datos
    logger.info("Cargando datos...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    calc = ServicioRestoCalculator(df)
    
    # Buscar un jugador conocido para ejemplo
    jugadores_unicos = pd.concat([df['winner_name'], df['loser_name']]).unique()
    jugador_ejemplo = None
    for nombre in ['Carlos Alcaraz', 'Novak Djokovic', 'Rafael Nadal', 'Roger Federer']:
        if nombre in jugadores_unicos:
            jugador_ejemplo = nombre
            break
    
    if jugador_ejemplo is None:
        jugador_ejemplo = jugadores_unicos[0]
    
    # Fecha de ejemplo (última fecha en el dataset)
    fecha_ejemplo = df['tourney_date'].max() - pd.Timedelta(days=30)
    
    # Estadísticas de servicio
    logger.info("=" * 60)
    logger.info(f"🎾 ESTADÍSTICAS DE SERVICIO - {jugador_ejemplo}")
    logger.info("=" * 60)
    
    serv = calc.calcular_estadisticas_servicio(
        jugador_nombre=jugador_ejemplo,
        fecha_partido=fecha_ejemplo
    )
    
    for key, value in serv.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.3f}")
        else:
            logger.info(f"   {key}: {value}")
    
    # Estadísticas de resto
    logger.info("\n" + "=" * 60)
    logger.info(f"🎯 ESTADÍSTICAS DE RESTO - {jugador_ejemplo}")
    logger.info("=" * 60)
    
    resto = calc.calcular_estadisticas_resto(
        jugador_nombre=jugador_ejemplo,
        fecha_partido=fecha_ejemplo
    )
    
    for key, value in resto.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.3f}")
        else:
            logger.info(f"   {key}: {value}")
    
    logger.info("\n✅ Proceso completado!")
