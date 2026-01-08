"""
Servicio de Generación de Features para API
============================================

Inicializa los calculadores de features una vez y los reutiliza
para todas las predicciones en la API.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from src.features.elo_rating_system import TennisELO
from src.features.features_servicio_resto import ServicioRestoCalculator
from src.features.features_forma_reciente import FormaRecienteCalculator
from src.features.features_h2h_mejorado import HeadToHeadCalculator
from src.features.features_superficie import SuperficieSpecializationCalculator

logger = logging.getLogger(__name__)


class FeatureGeneratorService:
    """
    Servicio singleton que mantiene los calculadores de features
    inicializados para generar predicciones rápidamente
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("🚀 Inicializando Feature Generator Service...")
        
        # Cargar datos históricos
        self.df_historico = self._cargar_datos_historicos()
        
        # Inicializar calculadores
        self._inicializar_calculadores()
        
        # Cargar features seleccionadas
        self.feature_cols = self._cargar_features_seleccionadas()
        
        self._initialized = True
        logger.info("✅ Feature Generator Service inicializado")
    
    def _cargar_datos_historicos(self):
        """Carga datos históricos de partidos"""
        try:
            # Intentar cargar desde TML
            df = pd.read_csv("datos/raw/atp_matches_2024_tml.csv")
            df['tourney_date'] = pd.to_datetime(df['tourney_date'])
            logger.info(f"✅ Datos históricos cargados: {len(df)} partidos")
            return df
        except Exception as e:
            logger.warning(f"⚠️  No se pudieron cargar datos históricos: {e}")
            logger.info("📝 Creando dataset mínimo...")
            # Dataset mínimo para que funcione
            return pd.DataFrame({
                'tourney_date': [datetime.now()],
                'winner_name': ['Unknown'],
                'loser_name': ['Unknown'],
                'surface': ['Hard'],
                'winner_rank': [100],
                'loser_rank': [100]
            })
    
    def _inicializar_calculadores(self):
        """Inicializa todos los calculadores de features"""
        logger.info("  📊 Inicializando calculadores...")
        
        # ELO
        self.elo_system = TennisELO(k_factor=32, base_rating=1500)
        self.df_historico = self.elo_system.calculate_historical_elos(self.df_historico)
        
        # Otros calculadores
        self.servicio_calc = ServicioRestoCalculator(self.df_historico)
        self.forma_calc = FormaRecienteCalculator(self.df_historico)
        self.h2h_calc = HeadToHeadCalculator(self.df_historico)
        self.superficie_calc = SuperficieSpecializationCalculator(self.df_historico)
        
        logger.info("  ✅ Calculadores inicializados")
    
    def _cargar_features_seleccionadas(self):
        """Carga la lista de features seleccionadas"""
        try:
            with open('resultados/selected_features.txt', 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            logger.info(f"  ✅ {len(features)} features cargadas desde archivo")
            return features
        except Exception as e:
            logger.warning(f"  ⚠️  No se pudo cargar selected_features.txt: {e}")
            logger.info("  📝 Usando lista de features por defecto (30 features)")
            # Lista hardcodeada de las 30 features del modelo
            return [
                'j1_jugador_elo', 'j1_oponente_elo', 'j1_elo_diff', 
                'j1_jugador_elo_surface', 'j1_oponente_elo_surface', 'j1_elo_diff_surface', 'j1_elo_expected_prob',
                'j1_jugador_rank', 'j1_oponente_rank', 'j1_rank_diff', 'j1_rank_ratio',
                'j1_diff_win_rate_60d',
                'j1_j1_serve_first_serve_in_pct', 'j1_j1_serve_first_serve_won_pct', 'j1_j1_serve_second_serve_won_pct',
                'j1_j1_serve_df_pct', 'j1_j1_serve_bp_saved_pct', 'j1_j1_serve_power',
                'j1_j2_serve_first_serve_in_pct', 'j1_j2_serve_first_serve_won_pct', 'j1_j2_serve_second_serve_won_pct',
                'j1_j2_serve_df_pct', 'j1_j2_serve_bp_saved_pct', 'j1_j2_serve_aces_per_match',
                'j1_j1_return_return_quality_score', 'j1_serve_vs_return_advantage',
                'j1_ventaja_superficie',
                'j1_rank_diff_x_forma', 'j1_elo_x_forma', 'j1_superficie_x_rank'
            ]
    
    def _obtener_ranking(self, jugador, fecha):
        """Obtiene el ranking más reciente de un jugador"""
        partidos = self.df_historico[
            ((self.df_historico['winner_name'] == jugador) | 
             (self.df_historico['loser_name'] == jugador)) &
            (self.df_historico['tourney_date'] < fecha)
        ].sort_values('tourney_date', ascending=False)
        
        if len(partidos) > 0:
            ultimo = partidos.iloc[0]
            if ultimo['winner_name'] == jugador:
                return ultimo.get('winner_rank', 999)
            else:
                return ultimo.get('loser_rank', 999)
        return 999
    
    def generar_features(self, jugador, oponente, superficie, fecha=None):
        """
        Genera las 30 features para un partido
        
        Args:
            jugador: Nombre del jugador
            oponente: Nombre del oponente
            superficie: Superficie (Hard/Clay/Grass)
            fecha: Fecha del partido (default: hoy)
            
        Returns:
            dict con features generadas
        """
        if fecha is None:
            fecha = datetime.now()
        
        features = {}
        
        # 1. ELO (7 features)
        elo_j = self.elo_system.get_rating(jugador)
        elo_o = self.elo_system.get_rating(oponente)
        elo_j_surf = self.elo_system.get_rating(jugador, superficie)
        elo_o_surf = self.elo_system.get_rating(oponente, superficie)
        
        features['jugador_elo'] = elo_j
        features['oponente_elo'] = elo_o
        features['elo_diff'] = elo_j - elo_o
        features['jugador_elo_surface'] = elo_j_surf
        features['oponente_elo_surface'] = elo_o_surf
        features['elo_diff_surface'] = elo_j_surf - elo_o_surf
        features['elo_expected_prob'] = self.elo_system.expected_score(elo_j, elo_o)
        
        # 2. Rankings (4 features)
        rank_j = self._obtener_ranking(jugador, fecha)
        rank_o = self._obtener_ranking(oponente, fecha)
        
        features['jugador_rank'] = rank_j
        features['oponente_rank'] = rank_o
        features['rank_diff'] = rank_o - rank_j
        features['rank_ratio'] = rank_j / max(rank_o, 1)
        
        # 3. Forma reciente (1 feature)
        forma_j = self.forma_calc.calcular_forma(jugador, fecha, ventana_dias=60)
        forma_o = self.forma_calc.calcular_forma(oponente, fecha, ventana_dias=60)
        
        features['diff_win_rate_60d'] = forma_j.get('win_rate_60d', 0.5) - forma_o.get('win_rate_60d', 0.5)
        
        # 4. Servicio y Resto (14 features)
        serv_j = self.servicio_calc.calcular_estadisticas_servicio(jugador, fecha)
        serv_o = self.servicio_calc.calcular_estadisticas_servicio(oponente, fecha)
        resto_j = self.servicio_calc.calcular_estadisticas_resto(jugador, fecha)
        
        features['j1_serve_first_serve_in_pct'] = serv_j.get('first_serve_in_pct', 0.6)
        features['j1_serve_first_serve_won_pct'] = serv_j.get('first_serve_won_pct', 0.7)
        features['j1_serve_second_serve_won_pct'] = serv_j.get('second_serve_won_pct', 0.5)
        features['j1_serve_df_pct'] = serv_j.get('df_pct', 0.03)
        features['j1_serve_bp_saved_pct'] = serv_j.get('bp_saved_pct', 0.6)
        features['j1_serve_power'] = serv_j.get('power', 0.7)
        
        features['j2_serve_first_serve_in_pct'] = serv_o.get('first_serve_in_pct', 0.6)
        features['j2_serve_first_serve_won_pct'] = serv_o.get('first_serve_won_pct', 0.7)
        features['j2_serve_second_serve_won_pct'] = serv_o.get('second_serve_won_pct', 0.5)
        features['j2_serve_df_pct'] = serv_o.get('df_pct', 0.03)
        features['j2_serve_bp_saved_pct'] = serv_o.get('bp_saved_pct', 0.6)
        features['j2_serve_aces_per_match'] = serv_o.get('aces_per_match', 5)
        
        features['j1_return_return_quality_score'] = resto_j.get('return_quality_score', 0.5)
        
        matchup = self.servicio_calc.calcular_matchup_servicio_resto(jugador, oponente, fecha)
        features['serve_vs_return_advantage'] = matchup.get('serve_vs_return_advantage', 0)
        
        # 5. Superficie (1 feature)
        try:
            ventaja_sup = self.superficie_calc.calcular_ventaja_superficie(jugador, oponente, fecha, superficie)
            features['ventaja_superficie'] = ventaja_sup.get('ventaja_superficie', 0)
        except:
            features['ventaja_superficie'] = 0
        
        # 6. Interacciones (3 features)
        features['rank_diff_x_forma'] = features['rank_diff'] * forma_j.get('win_rate_60d', 0.5)
        features['elo_x_forma'] = features['elo_diff'] * features['diff_win_rate_60d']
        features['superficie_x_rank'] = features['ventaja_superficie'] * features['rank_diff']
        
        return features
    
    def generar_features_bidireccionales(self, jugador1, jugador2, superficie, fecha=None):
        """
        Genera features bidireccionales (j1 y j2) como espera el modelo
        
        Returns:
            dict con features en formato j1_* y j2_*
        """
        # Generar features para jugador1
        features_j1 = self.generar_features(jugador1, jugador2, superficie, fecha)
        
        # Generar features para jugador2
        features_j2 = self.generar_features(jugador2, jugador1, superficie, fecha)
        
        # Combinar con prefijos
        features_combined = {}
        for key, value in features_j1.items():
            features_combined[f'j1_{key}'] = value
        for key, value in features_j2.items():
            features_combined[f'j2_{key}'] = value
        
        return features_combined
