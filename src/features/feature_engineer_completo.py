"""
Feature Engineering Completo para Fase 3
Integra todos los calculadores de features avanzadas
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Añadir path para imports
sys.path.append(str(Path(__file__).parent))

from elo_rating_system import TennisELO
from features_servicio_resto import ServicioRestoCalculator
from features_fatiga import FatigaCalculator
from features_forma_reciente import FormaRecienteCalculator
from features_h2h_mejorado import HeadToHeadCalculator
from features_superficie import SuperficieSpecializationCalculator
from features_momentum import MomentumCalculator, crear_features_momentum_partido
from features_tournament_context import TournamentContextCalculator, crear_features_torneo_partido

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteFeatureEngineer:
    """
    Pipeline completo de feature engineering con todas las features de Fase 3
    
    Incluye:
    - ELO Rating System (general y por superficie)
    - Estadísticas de servicio y resto
    - Métricas de fatiga
    - Forma reciente
    - Head-to-Head mejorado
    - Especialización por superficie
    - Features de momentum (tendencias de forma)
    - Features de contexto de torneo
    - Features de interacción
    """
    
    def __init__(self, df_partidos, elo_system=None):
        """
        Args:
            df_partidos: DataFrame con todos los partidos históricos
            elo_system: Sistema ELO pre-calculado (opcional)
        """
        self.df = df_partidos.copy()
        self.df['tourney_date'] = pd.to_datetime(self.df['tourney_date'])
        
        # Inicializar calculadores
        logger.info("Inicializando calculadores de features...")
        
        # ELO System
        if elo_system is None:
            logger.info("  Creando sistema ELO...")
            self.elo_system = TennisELO(k_factor=32, base_rating=1500)
            logger.info("  Calculando ELOs históricos...")
            self.df = self.elo_system.calculate_historical_elos(self.df)
        else:
            self.elo_system = elo_system
        
        # Otros calculadores
        self.servicio_calc = ServicioRestoCalculator(self.df)
        self.fatiga_calc = FatigaCalculator(self.df)
        self.forma_calc = FormaRecienteCalculator(self.df)
        self.h2h_calc = HeadToHeadCalculator(self.df)
        self.superficie_calc = SuperficieSpecializationCalculator(self.df)
        self.momentum_calc = MomentumCalculator(self.df)
        self.tournament_calc = TournamentContextCalculator(self.df)
        
        logger.info("✅ Calculadores inicializados")
    
    def crear_features_partido(self, partido_row):
        """
        Crea TODAS las features para un partido
        
        Args:
            partido_row: Serie con datos del partido
                Requerido: fecha, jugador_nombre, oponente_nombre, 
                          jugador_rank, oponente_rank, superficie
        
        Returns:
            dict con todas las features
        """
        
        features = {}
        
        # 1. Features básicas (Fase 1)
        features.update(self._features_basicas(partido_row))
        
        # 2. Features de ELO
        features.update(self._features_elo(partido_row))
        
        # 3. Features de forma reciente
        forma_j1 = self.forma_calc.calcular_forma(
            partido_row['jugador_nombre'],
            partido_row['fecha'],
            ventana_dias=60
        )
        forma_j2 = self.forma_calc.calcular_forma(
            partido_row['oponente_nombre'],
            partido_row['fecha'],
            ventana_dias=60
        )
        
        # Añadir con prefijos
        for key, value in forma_j1.items():
            features[f'j1_{key}'] = value
        for key, value in forma_j2.items():
            features[f'j2_{key}'] = value
        
        # Features de diferencia de forma
        features['diff_win_rate_60d'] = forma_j1['win_rate_60d'] - forma_j2['win_rate_60d']
        features['diff_racha'] = forma_j1['racha_actual'] - forma_j2['racha_actual']
        
        # 4. Features H2H
        h2h = self.h2h_calc.calcular_h2h(
            partido_row['jugador_nombre'],
            partido_row['oponente_nombre'],
            partido_row['fecha'],
            superficie=partido_row['superficie']
        )
        features.update(h2h)
        
        # 5. Features de servicio y resto
        serv_j1 = self.servicio_calc.calcular_estadisticas_servicio(
            partido_row['jugador_nombre'],
            partido_row['fecha']
        )
        resto_j1 = self.servicio_calc.calcular_estadisticas_resto(
            partido_row['jugador_nombre'],
            partido_row['fecha']
        )
        serv_j2 = self.servicio_calc.calcular_estadisticas_servicio(
            partido_row['oponente_nombre'],
            partido_row['fecha']
        )
        resto_j2 = self.servicio_calc.calcular_estadisticas_resto(
            partido_row['oponente_nombre'],
            partido_row['fecha']
        )
        
        # Añadir con prefijos
        for key, value in serv_j1.items():
            if key != 'n_partidos':  # Evitar duplicados
                features[f'j1_serve_{key}'] = value
        for key, value in resto_j1.items():
            if key != 'n_partidos':
                features[f'j1_return_{key}'] = value
        for key, value in serv_j2.items():
            if key != 'n_partidos':
                features[f'j2_serve_{key}'] = value
        for key, value in resto_j2.items():
            if key != 'n_partidos':
                features[f'j2_return_{key}'] = value
        
        # Matchup servicio vs resto
        matchup = self.servicio_calc.calcular_matchup_servicio_resto(
            partido_row['jugador_nombre'],
            partido_row['oponente_nombre'],
            partido_row['fecha']
        )
        features.update(matchup)
        
        # 6. Features de fatiga
        fatiga_j1 = self.fatiga_calc.calcular_fatiga(
            partido_row['jugador_nombre'],
            partido_row['fecha']
        )
        fatiga_j2 = self.fatiga_calc.calcular_fatiga(
            partido_row['oponente_nombre'],
            partido_row['fecha']
        )
        
        # Añadir con prefijos
        for key, value in fatiga_j1.items():
            features[f'j1_fatiga_{key}'] = value
        for key, value in fatiga_j2.items():
            features[f'j2_fatiga_{key}'] = value
        
        # Ventaja fatiga
        ventaja_fatiga = self.fatiga_calc.calcular_ventaja_fatiga(
            partido_row['jugador_nombre'],
            partido_row['oponente_nombre'],
            partido_row['fecha']
        )
        features.update(ventaja_fatiga)
        
        # 7. Features de superficie
        ventaja_sup = self.superficie_calc.calcular_ventaja_superficie(
            partido_row['jugador_nombre'],
            partido_row['oponente_nombre'],
            partido_row['fecha'],
            partido_row['superficie']
        )
        features.update(ventaja_sup)
        
        # 8. Features de momentum
        momentum_features = crear_features_momentum_partido(self.momentum_calc, partido_row)
        features.update(momentum_features)
        
        # 9. Features de contexto de torneo
        tournament_features = crear_features_torneo_partido(self.tournament_calc, partido_row)
        features.update(tournament_features)
        
        # 10. Features de interacción
        features.update(self._features_interaccion(features))
        
        return features
    
    def _features_basicas(self, partido):
        """Features de Fase 1"""
        return {
            'jugador_rank': partido['jugador_rank'],
            'oponente_rank': partido['oponente_rank'],
            'rank_diff': partido['oponente_rank'] - partido['jugador_rank'],
            'rank_ratio': partido['jugador_rank'] / max(partido['oponente_rank'], 1),
            'jugador_top10': 1 if partido['jugador_rank'] <= 10 else 0,
            'oponente_top10': 1 if partido['oponente_rank'] <= 10 else 0,
            'jugador_top50': 1 if partido['jugador_rank'] <= 50 else 0,
            'oponente_top50': 1 if partido['oponente_rank'] <= 50 else 0,
            'surface_hard': 1 if partido['superficie'] == 'Hard' else 0,
            'surface_clay': 1 if partido['superficie'] == 'Clay' else 0,
            'surface_grass': 1 if partido['superficie'] == 'Grass' else 0
        }
    
    def _features_elo(self, partido):
        """Features de ELO"""
        elo_j1 = self.elo_system.get_rating(partido['jugador_nombre'])
        elo_j2 = self.elo_system.get_rating(partido['oponente_nombre'])
        
        superficie = partido['superficie']
        elo_j1_surf = self.elo_system.get_rating(partido['jugador_nombre'], superficie)
        elo_j2_surf = self.elo_system.get_rating(partido['oponente_nombre'], superficie)
        
        return {
            'jugador_elo': elo_j1,
            'oponente_elo': elo_j2,
            'elo_diff': elo_j1 - elo_j2,
            'jugador_elo_surface': elo_j1_surf,
            'oponente_elo_surface': elo_j2_surf,
            'elo_diff_surface': elo_j1_surf - elo_j2_surf,
            'elo_expected_prob': self.elo_system.expected_score(elo_j1, elo_j2)
        }
    
    def _features_interaccion(self, features):
        """
        Features de interacción entre otras features
        """
        interacciones = {}
        
        # Interacción rank × forma
        if 'j1_win_rate_60d' in features and 'rank_diff' in features:
            interacciones['rank_diff_x_forma'] = (
                features['rank_diff'] * features['j1_win_rate_60d']
            )
        
        # Interacción H2H × forma actual
        if 'h2h_win_rate' in features and 'j1_win_rate_60d' in features:
            interacciones['h2h_x_forma_actual'] = (
                features['h2h_win_rate'] * features['j1_win_rate_60d']
            )
        
        # Ventaja superficie × ranking
        if 'ventaja_superficie' in features and 'rank_diff' in features:
            interacciones['superficie_x_rank'] = (
                features['ventaja_superficie'] * features['rank_diff']
            )
        
        # ELO × forma reciente
        if 'elo_diff' in features and 'diff_win_rate_60d' in features:
            interacciones['elo_x_forma'] = (
                features['elo_diff'] * features['diff_win_rate_60d']
            )
        
        return interacciones
    
    def procesar_dataset_completo(self, save_path=None):
        """
        Procesa todo el dataset con las features completas
        
        Args:
            save_path: Ruta donde guardar el dataset procesado
            
        Returns:
            DataFrame con todas las features
        """
        
        logger.info("=" * 60)
        logger.info("🔧 PROCESANDO DATASET COMPLETO CON TODAS LAS FEATURES")
        logger.info("=" * 60)
        
        # Primero crear dataset de jugadores (formato de Fase 1)
        logger.info("\n📊 Creando dataset de jugadores...")
        datos = []
        
        for idx, row in self.df.iterrows():
            # Partido original: Ganador vs Perdedor (resultado = 1)
            partido_1 = {
                'fecha': row['tourney_date'],
                'jugador_nombre': row['winner_name'],
                'oponente_nombre': row['loser_name'],
                'jugador_rank': row['winner_rank'],
                'oponente_rank': row['loser_rank'],
                'superficie': row['surface'],
                'tourney_level': row.get('tourney_level', 'D'),
                'tourney_name': row.get('tourney_name', ''),
                'round': row.get('round', ''),
                'resultado': 1
            }
            
            # Partido invertido: Perdedor vs Ganador (resultado = 0)
            partido_2 = {
                'fecha': row['tourney_date'],
                'jugador_nombre': row['loser_name'],
                'oponente_nombre': row['winner_name'],
                'jugador_rank': row['loser_rank'],
                'oponente_rank': row['winner_rank'],
                'superficie': row['surface'],
                'tourney_level': row.get('tourney_level', 'D'),
                'tourney_name': row.get('tourney_name', ''),
                'round': row.get('round', ''),
                'resultado': 0
            }
            
            datos.append(partido_1)
            datos.append(partido_2)
        
        df_jugadores = pd.DataFrame(datos)
        logger.info(f"✅ Dataset de jugadores creado: {len(df_jugadores):,} filas")
        
        # Ahora crear features para cada partido
        logger.info("\n🔧 Creando features avanzadas...")
        features_list = []
        targets = []
        
        total = len(df_jugadores)
        for idx, row in df_jugadores.iterrows():
            if idx % 500 == 0:
                logger.info(f"   Progreso: {idx:,}/{total:,} ({idx/total*100:.1f}%)")
            
            try:
                features = self.crear_features_partido(row)
                features_list.append(features)
                targets.append(row['resultado'])
            except Exception as e:
                logger.warning(f"   ⚠️  Error en partido {idx}: {e}")
                continue
        
        # Crear DataFrame
        df_features = pd.DataFrame(features_list)
        df_features['resultado'] = targets
        df_features['fecha'] = df_jugadores['fecha'].values[:len(df_features)]
        
        logger.info(f"\n✅ Procesamiento completado!")
        logger.info(f"   Features creadas: {len(df_features.columns) - 2}")
        logger.info(f"   Partidos procesados: {len(df_features):,}")
        
        # Mostrar algunas features
        logger.info(f"\n📊 Primeras 20 features:")
        for col in sorted(df_features.columns)[:20]:
            if col not in ['resultado', 'fecha']:
                logger.info(f"   - {col}")
        
        # Guardar
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df_features.to_csv(save_path, index=False)
            logger.info(f"\n💾 Dataset guardado: {save_path}")
        
        return df_features


# Ejecutar
if __name__ == "__main__":
    # Cargar datos limpios
    logger.info("Cargando datos limpios...")
    df = pd.read_csv("datos/processed/atp_matches_clean.csv")
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    
    logger.info(f"📊 Partidos cargados: {len(df):,}")
    logger.info(f"📅 Rango: {df['tourney_date'].min()} - {df['tourney_date'].max()}")
    
    # Crear feature engineer completo
    logger.info("\n" + "=" * 60)
    logger.info("🚀 INICIANDO FEATURE ENGINEERING COMPLETO")
    logger.info("=" * 60)
    
    engineer = CompleteFeatureEngineer(df)
    
    # Procesar todo
    df_features_completo = engineer.procesar_dataset_completo(
        save_path="datos/processed/dataset_features_fase3_completas.csv"
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ FEATURE ENGINEERING FASE 3 COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"📊 Total de features: {len(df_features_completo.columns) - 2}")
    logger.info(f"📊 Total de filas: {len(df_features_completo):,}")
    
    # Mostrar todas las features
    logger.info("\n📋 LISTA COMPLETA DE FEATURES:")
    for i, col in enumerate(sorted(df_features_completo.columns), 1):
        if col not in ['resultado', 'fecha']:
            logger.info(f"   {i}. {col}")
