"""
Servicio de Generación de Features para API
============================================

Inicializa los calculadores de features una vez y los reutiliza
para todas las predicciones en la API.
"""

import pandas as pd
from datetime import datetime
import logging

from src.features.elo_rating_system import TennisELO
from src.features.features_servicio_resto import ServicioRestoCalculator
from src.features.features_forma_reciente import FormaRecienteCalculator
from src.features.features_h2h_mejorado import HeadToHeadCalculator
from src.features.features_superficie import SuperficieSpecializationCalculator
from src.utils.player_name_normalizer import PlayerNameNormalizer

logger = logging.getLogger(__name__)


class FeatureGeneratorService:
    """
    Servicio que mantiene los calculadores de features
    inicializados para generar predicciones rápidamente
    """

    def __init__(self):
        """Inicializa el servicio cargando datos históricos"""
        logger.info("🚀 Inicializando Feature Generator Service...")

        # Cargar datos históricos
        self.df_historico = self._cargar_datos_historicos()

        # Inicializar calculadores
        self._inicializar_calculadores()

        # Cargar features seleccionadas
        self.feature_cols = self._cargar_features_seleccionadas()
        
        # Inicializar normalizador de nombres
        self._inicializar_normalizador()

        logger.info("✅ Feature Generator Service inicializado")

    def _cargar_datos_historicos(self):
        """Carga datos históricos de partidos de múltiples años"""
        try:
            # Cargar datos de 2022-2026 para tener suficiente histórico
            dfs = []
            años = [2022, 2023, 2024, 2025, 2026]
            
            logger.info("📂 Cargando datos históricos...")
            for año in años:
                try:
                    file_path = f"datos/raw/atp_matches_{año}_tml.csv"
                    df_año = pd.read_csv(file_path)
                    df_año["tourney_date"] = pd.to_datetime(df_año["tourney_date"])
                    dfs.append(df_año)
                    logger.info(f"  ✅ Cargados {len(df_año)} partidos de {año}")
                except FileNotFoundError:
                    logger.warning(f"  ⚠️  Archivo no encontrado: datos/raw/atp_matches_{año}_tml.csv")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error cargando {año}: {e}")
            
            if not dfs:
                raise Exception("❌ No se pudo cargar ningún archivo de datos históricos")
            
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values("tourney_date").reset_index(drop=True)
            
            # Validación de datos
            logger.info(f"✅ Total datos históricos: {len(df)} partidos ({años[0]}-{años[-1]})")
            
            # Verificar que tenemos datos recientes
            fecha_mas_reciente = df["tourney_date"].max()
            fecha_mas_antigua = df["tourney_date"].min()
            logger.info(f"📅 Rango de fechas: {fecha_mas_antigua.date()} a {fecha_mas_reciente.date()}")
            
            # Contar jugadores únicos
            jugadores_unicos = set(df["winner_name"].unique()) | set(df["loser_name"].unique())
            logger.info(f"👥 Jugadores únicos en histórico: {len(jugadores_unicos)}")
            
            # Validar que tenemos suficientes datos
            if len(df) < 1000:
                logger.warning(f"⚠️  ADVERTENCIA: Solo {len(df)} partidos en histórico (recomendado: >1000)")
            
            if len(jugadores_unicos) < 100:
                logger.warning(f"⚠️  ADVERTENCIA: Solo {len(jugadores_unicos)} jugadores únicos (recomendado: >100)")
            
            return df
                
        except Exception as e:
            logger.error(f"❌ ERROR CRÍTICO cargando datos históricos: {e}")
            logger.error("❌ El sistema NO PUEDE generar predicciones precisas sin datos históricos")
            logger.error("❌ Las predicciones serán UNIFORMES y NO CONFIABLES")
            raise Exception(f"No se pudieron cargar datos históricos: {e}")

    def _inicializar_calculadores(self):
        """Inicializa todos los calculadores de features"""
        logger.info("  📊 Inicializando calculadores...")

        # ELO
        self.elo_system = TennisELO(k_factor=32, base_rating=1500)
        self.df_historico = self.elo_system.calculate_historical_elos(self.df_historico)
        logger.info(f"  ✅ Sistema ELO inicializado con {len(self.elo_system.ratings)} jugadores")

        # Otros calculadores
        self.servicio_calc = ServicioRestoCalculator(self.df_historico)
        self.forma_calc = FormaRecienteCalculator(self.df_historico)
        self.h2h_calc = HeadToHeadCalculator(self.df_historico)
        self.superficie_calc = SuperficieSpecializationCalculator(self.df_historico)

        logger.info("  ✅ Calculadores inicializados")
    
    def _inicializar_normalizador(self):
        """Inicializa el normalizador de nombres de jugadores"""
        # Obtener todos los nombres únicos del histórico
        jugadores_historico = set(
            list(self.df_historico["winner_name"].unique()) +
            list(self.df_historico["loser_name"].unique())
        )
        
        self.name_normalizer = PlayerNameNormalizer(jugadores_historico)
        self.known_players = jugadores_historico  # Guardar para detección de confianza
        logger.info(f"  ✅ Normalizador de nombres inicializado")

    def _cargar_features_seleccionadas(self):
        """Carga la lista de features seleccionadas"""
        try:
            with open("resultados/selected_features.txt", "r") as f:
                features = [line.strip() for line in f if line.strip()]
            logger.info(f"  ✅ {len(features)} features cargadas desde archivo")
            return features
        except Exception as e:
            logger.warning(f"  ⚠️  No se pudo cargar selected_features.txt: {e}")
            logger.info("  📝 Usando lista de features por defecto (30 features)")
            # Lista hardcodeada de las 30 features del modelo
            return [
                "j1_jugador_elo",
                "j1_oponente_elo",
                "j1_elo_diff",
                "j1_jugador_elo_surface",
                "j1_oponente_elo_surface",
                "j1_elo_diff_surface",
                "j1_elo_expected_prob",
                "j1_jugador_rank",
                "j1_oponente_rank",
                "j1_rank_diff",
                "j1_rank_ratio",
                "j1_diff_win_rate_60d",
                "j1_j1_serve_first_serve_in_pct",
                "j1_j1_serve_first_serve_won_pct",
                "j1_j1_serve_second_serve_won_pct",
                "j1_j1_serve_df_pct",
                "j1_j1_serve_bp_saved_pct",
                "j1_j1_serve_power",
                "j1_j2_serve_first_serve_in_pct",
                "j1_j2_serve_first_serve_won_pct",
                "j1_j2_serve_second_serve_won_pct",
                "j1_j2_serve_df_pct",
                "j1_j2_serve_bp_saved_pct",
                "j1_j2_serve_aces_per_match",
                "j1_j1_return_return_quality_score",
                "j1_serve_vs_return_advantage",
                "j1_ventaja_superficie",
                "j1_rank_diff_x_forma",
                "j1_elo_x_forma",
                "j1_superficie_x_rank",
            ]

    def _obtener_ranking(self, jugador, fecha):
        """Obtiene el ranking más reciente de un jugador"""
        partidos = self.df_historico[
            (
                (self.df_historico["winner_name"] == jugador)
                | (self.df_historico["loser_name"] == jugador)
            )
            & (self.df_historico["tourney_date"] < fecha)
        ].sort_values("tourney_date", ascending=False)

        if len(partidos) > 0:
            ultimo = partidos.iloc[0]
            if ultimo["winner_name"] == jugador:
                return ultimo.get("winner_rank", 999)
            else:
                return ultimo.get("loser_rank", 999)
        return 999

    def generar_features(self, jugador, oponente, superficie, fecha=None):
        """
        Genera las 30 features para un partido

        Args:
            jugador: Nombre del jugador (puede ser nombre abreviado de API)
            oponente: Nombre del oponente (puede ser nombre abreviado de API)
            superficie: Superficie (Hard/Clay/Grass)
            fecha: Fecha del partido (default: hoy)

        Returns:
            dict con features generadas
        """
        if fecha is None:
            fecha = datetime.now()
        
        # Normalizar nombres de jugadores para que coincidan con histórico
        jugador_normalizado = self.name_normalizer.normalize(jugador)
        oponente_normalizado = self.name_normalizer.normalize(oponente)
        
        if jugador_normalizado != jugador or oponente_normalizado != oponente:
            logger.debug(
                f"🔍 Nombres normalizados: '{jugador}' → '{jugador_normalizado}', "
                f"'{oponente}' → '{oponente_normalizado}'"
            )
        
        # Detectar si jugadores están en datos históricos
        jugador_conocido = jugador_normalizado in self.known_players
        oponente_conocido = oponente_normalizado in self.known_players
        
        # Calcular nivel de confianza
        if jugador_conocido and oponente_conocido:
            confidence_level = "HIGH"
            confidence_score = 1.0
        elif jugador_conocido or oponente_conocido:
            confidence_level = "MEDIUM"
            confidence_score = 0.5
        else:
            confidence_level = "LOW"
            confidence_score = 0.0
            logger.warning(
                f"⚠️  BAJA CONFIANZA: Jugadores sin historial - {jugador_normalizado} vs {oponente_normalizado}"
            )

        logger.debug(
            f"🎾 Generando features para: {jugador_normalizado} vs {oponente_normalizado} en {superficie} "
            f"(Confianza: {confidence_level})"
        )
        features = {}

        # 1. ELO (7 features)
        elo_j = self.elo_system.get_rating(jugador_normalizado)
        elo_o = self.elo_system.get_rating(oponente_normalizado)
        elo_j_surf = self.elo_system.get_rating(jugador_normalizado, superficie)
        elo_o_surf = self.elo_system.get_rating(oponente_normalizado, superficie)

        features["jugador_elo"] = elo_j
        features["oponente_elo"] = elo_o
        features["elo_diff"] = elo_j - elo_o
        features["jugador_elo_surface"] = elo_j_surf
        features["oponente_elo_surface"] = elo_o_surf
        features["elo_diff_surface"] = elo_j_surf - elo_o_surf
        features["elo_expected_prob"] = self.elo_system.expected_score(elo_j, elo_o)
        
        logger.debug(f"  ELO: {jugador_normalizado}={elo_j:.0f} vs {oponente_normalizado}={elo_o:.0f} (diff={elo_j-elo_o:.0f})")

        # 2. Rankings (4 features)
        rank_j = self._obtener_ranking(jugador_normalizado, fecha)
        rank_o = self._obtener_ranking(oponente_normalizado, fecha)

        features["jugador_rank"] = rank_j
        features["oponente_rank"] = rank_o
        features["rank_diff"] = rank_o - rank_j
        features["rank_ratio"] = rank_j / max(rank_o, 1)

        # 3. Forma reciente (1 feature)
        forma_j = self.forma_calc.calcular_forma(jugador_normalizado, fecha, ventana_dias=60)
        forma_o = self.forma_calc.calcular_forma(oponente_normalizado, fecha, ventana_dias=60)

        features["diff_win_rate_60d"] = forma_j.get("win_rate_60d", 0.5) - forma_o.get(
            "win_rate_60d", 0.5
        )

        # 4. Servicio y Resto (14 features)
        serv_j = self.servicio_calc.calcular_estadisticas_servicio(jugador_normalizado, fecha)
        serv_o = self.servicio_calc.calcular_estadisticas_servicio(oponente_normalizado, fecha)
        resto_j = self.servicio_calc.calcular_estadisticas_resto(jugador_normalizado, fecha)

        features["j1_serve_first_serve_in_pct"] = serv_j.get("first_serve_in_pct", 0.6)
        features["j1_serve_first_serve_won_pct"] = serv_j.get("first_serve_won_pct", 0.7)
        features["j1_serve_second_serve_won_pct"] = serv_j.get("second_serve_won_pct", 0.5)
        features["j1_serve_df_pct"] = serv_j.get("df_pct", 0.03)
        features["j1_serve_bp_saved_pct"] = serv_j.get("bp_saved_pct", 0.6)
        features["j1_serve_power"] = serv_j.get("power", 0.7)

        features["j2_serve_first_serve_in_pct"] = serv_o.get("first_serve_in_pct", 0.6)
        features["j2_serve_first_serve_won_pct"] = serv_o.get("first_serve_won_pct", 0.7)
        features["j2_serve_second_serve_won_pct"] = serv_o.get("second_serve_won_pct", 0.5)
        features["j2_serve_df_pct"] = serv_o.get("df_pct", 0.03)
        features["j2_serve_bp_saved_pct"] = serv_o.get("bp_saved_pct", 0.6)
        features["j2_serve_aces_per_match"] = serv_o.get("aces_per_match", 5)

        features["j1_return_return_quality_score"] = resto_j.get("return_quality_score", 0.5)

        matchup = self.servicio_calc.calcular_matchup_servicio_resto(jugador_normalizado, oponente_normalizado, fecha)
        features["serve_vs_return_advantage"] = matchup.get("serve_vs_return_advantage", 0)

        # 5. Superficie (1 feature)
        try:
            ventaja_sup = self.superficie_calc.calcular_ventaja_superficie(
                jugador_normalizado, oponente_normalizado, fecha, superficie
            )
            features["ventaja_superficie"] = ventaja_sup.get("ventaja_superficie", 0)
        except:
            features["ventaja_superficie"] = 0

        # 6. Interacciones (3 features)
        features["rank_diff_x_forma"] = features["rank_diff"] * forma_j.get("win_rate_60d", 0.5)
        features["elo_x_forma"] = features["elo_diff"] * features["diff_win_rate_60d"]
        features["superficie_x_rank"] = features["ventaja_superficie"] * features["rank_diff"]
        
        # 7. Metadatos de confianza (NO son features del modelo, son metadata)
        features["_confidence_level"] = confidence_level
        features["_confidence_score"] = confidence_score
        features["_player1_known"] = jugador_conocido
        features["_player2_known"] = oponente_conocido

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
            features_combined[f"j1_{key}"] = value
        for key, value in features_j2.items():
            features_combined[f"j2_{key}"] = value

        return features_combined
