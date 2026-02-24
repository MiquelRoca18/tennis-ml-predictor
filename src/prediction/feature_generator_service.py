"""
Servicio de Generación de Features para API
============================================

Inicializa los calculadores de features una vez y los reutiliza
para todas las predicciones en la API.
"""

import pandas as pd
from datetime import datetime, date
import logging

# Años a cargar para ELO: desde (año_actual - AÑOS_ELO_ATRAS) hasta año_actual.
# Así, al empezar 2027 no hay que tocar código: se buscará 2027.csv automáticamente.
AÑOS_ELO_ATRAS = 8  # ej. en 2026 cargamos 2018..2026

from src.features.elo_rating_system import TennisELO
from src.features.features_servicio_resto import ServicioRestoCalculator
from src.features.features_forma_reciente import FormaRecienteCalculator
from src.features.features_h2h_mejorado import HeadToHeadCalculator
from src.features.features_superficie import SuperficieSpecializationCalculator
from src.utils.player_name_normalizer import PlayerNameNormalizer

logger = logging.getLogger(__name__)

# Singleton (igual que backtesting: un único estado que se actualiza con cada partido)
_instance = None


def get_instance() -> "FeatureGeneratorService":
    """Devuelve la instancia singleton del servicio (crea si no existe)."""
    global _instance
    if _instance is None:
        _instance = FeatureGeneratorService()
    return _instance


def reset_instance() -> None:
    """Borra la instancia singleton. La próxima get_instance() recargará desde CSV/BD (útil tras actualizar datos ELO)."""
    global _instance
    _instance = None
    logger.info("🔄 FeatureGeneratorService reseteado - se recargará en próxima predicción")


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

        # feature_cols ya no se usa (sistema en modo baseline ELO + mercado, sin modelo ML)
        self.feature_cols = []

        # Inicializar normalizador de nombres
        self._inicializar_normalizador()

        logger.info("✅ Feature Generator Service inicializado")

    def _cargar_datos_historicos(self):
        """
        Carga datos históricos para ELO.

        Orden de preferencia:
        1. CSV en datos/raw/ (atp_matches_*_tml.csv o 2022.csv, 2023.csv, ...) si existen.
           Así en Railway puedes incluir los CSV en la imagen y no hace falta importar a la BD.
        2. Si no hay CSV, desde la BD (matches WHERE estado = 'completado').
        3. Si la BD está vacía, fallback a CSV por si acaso.
        """
        import os
        import glob

        # 1) Intentar primero CSV si hay archivos en datos/raw/ (típico en Railway con CSVs en imagen)
        df_csv = self._cargar_datos_historicos_desde_csv()
        if df_csv is not None and len(df_csv) > 0:
            logger.info(f"✅ Datos históricos desde CSV: {len(df_csv)} partidos (no hace falta importar a BD)")
            # Opcional: añadir partidos completados recientes de la BD que no estén en el CSV
            df_db_extra = self._cargar_completados_db_como_dataframe()
            if df_db_extra is not None and len(df_db_extra) > 0:
                df_csv = pd.concat([df_csv, df_db_extra], ignore_index=True)
                df_csv = df_csv.drop_duplicates(subset=["tourney_date", "winner_name", "loser_name"], keep="last")
                df_csv = df_csv.sort_values("tourney_date").reset_index(drop=True)
                logger.info(f"   + {len(df_db_extra)} partidos completados desde BD (total {len(df_csv)})")
            return df_csv

        # 2) Sin CSV útiles: cargar desde BD
        try:
            logger.info("📂 Cargando datos históricos desde Base de Datos...")
            query = """
            SELECT 
                fecha_partido, jugador1_nombre, jugador2_nombre, resultado_ganador,
                superficie, torneo, tournament_season, jugador1_ranking, jugador2_ranking, resultado_marcador
            FROM matches 
            WHERE estado = 'completado' AND resultado_ganador IS NOT NULL
            ORDER BY fecha_partido ASC
            """
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                df_db = self._load_from_postgres(database_url, query)
            else:
                df_db = self._load_from_sqlite(query)

            if df_db.empty:
                logger.warning("⚠️  La base de datos está vacía o no tiene partidos completados.")
                return self._cargar_datos_historicos_csv_fallback()

            df = self._db_rows_to_legacy_df(df_db)
            logger.info(f"✅ Total datos históricos desde DB: {len(df)} partidos")
            return df
        except Exception as e:
            logger.error(f"❌ Error cargando datos de DB: {e}")
            return self._cargar_datos_historicos_csv_fallback()

    def _cargar_datos_historicos_desde_csv(self):
        """
        Carga partidos desde datos/raw/ (TML: atp_matches_*.csv o 2022.csv, 2023.csv, ...).
        Devuelve DataFrame o None si no hay archivos válidos.
        """
        import os
        import glob
        csv_files = []
        data_path = os.path.join("datos", "raw")
        if not os.path.isdir(data_path):
            return None
        for path in glob.glob(os.path.join(data_path, "atp_matches_*.csv")):
            csv_files.append(path)
        current_year = date.today().year
        for year in range(current_year - AÑOS_ELO_ATRAS, current_year + 1):
            if year < 2018:
                continue
            p = os.path.join(data_path, f"{year}.csv")
            if p not in csv_files and os.path.isfile(p):
                csv_files.append(p)
        csv_files.sort()
        if not csv_files:
            return None
        dfs = []
        for path in csv_files:
            try:
                df_a = pd.read_csv(path)
                if "tourney_date" not in df_a.columns or "winner_name" not in df_a.columns:
                    continue
                df_a["tourney_date"] = pd.to_datetime(df_a["tourney_date"], errors="coerce")
                df_a = df_a.dropna(subset=["tourney_date"])
                dfs.append(df_a)
            except Exception:
                continue
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("tourney_date").reset_index(drop=True)
        return df

    def _cargar_completados_db_como_dataframe(self):
        """Carga partidos completados de la BD en formato legacy (para mezclar con CSV)."""
        import os
        try:
            query = """
            SELECT fecha_partido, jugador1_nombre, jugador2_nombre, resultado_ganador,
                   superficie, torneo, tournament_season, jugador1_ranking, jugador2_ranking, resultado_marcador
            FROM matches WHERE estado = 'completado' AND resultado_ganador IS NOT NULL
            ORDER BY fecha_partido ASC
            """
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                df_db = self._load_from_postgres(database_url, query)
            else:
                df_db = self._load_from_sqlite(query)
            if df_db.empty:
                return None
            return self._db_rows_to_legacy_df(df_db)
        except Exception:
            return None

    def _db_rows_to_legacy_df(self, df_db: pd.DataFrame) -> pd.DataFrame:
        """Convierte filas de la tabla matches al formato legacy (winner_name, loser_name, tourney_date, ...)."""
        rows = []
        for _, row in df_db.iterrows():
            try:
                winner = row["resultado_ganador"]
                j1, j2 = row["jugador1_nombre"], row["jugador2_nombre"]
                if winner == j1:
                    winner_name, loser_name = j1, j2
                    winner_rank = row.get("jugador1_ranking") or 0
                    loser_rank = row.get("jugador2_ranking") or 0
                else:
                    winner_name, loser_name = j2, j1
                    winner_rank = row.get("jugador2_ranking") or 0
                    loser_rank = row.get("jugador1_ranking") or 0
                rows.append({
                    "tourney_date": pd.to_datetime(row["fecha_partido"]),
                    "tourney_name": row.get("torneo", ""),
                    "surface": row.get("superficie", "Hard"),
                    "winner_name": winner_name,
                    "loser_name": loser_name,
                    "winner_rank": winner_rank,
                    "loser_rank": loser_rank,
                    "score": str(row.get("resultado_marcador") or ""),
                })
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=["tourney_date", "winner_name", "loser_name", "surface"])
        df = pd.DataFrame(rows)
        df = df.sort_values("tourney_date").reset_index(drop=True)
        return df

    def _load_from_postgres(self, database_url: str, query: str) -> pd.DataFrame:
        """Carga datos desde PostgreSQL"""
        from sqlalchemy import create_engine, text
        
        # Fix Railway's postgres:// to postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        
        logger.info("🐘 Conectando a PostgreSQL para datos históricos...")
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        
        logger.info(f"✅ Datos cargados desde PostgreSQL: {len(df)} partidos")
        return df

    def _load_from_sqlite(self, query: str) -> pd.DataFrame:
        """Carga datos desde SQLite"""
        import sqlite3
        
        logger.info("📂 Conectando a SQLite para datos históricos...")
        conn = sqlite3.connect("matches_v2.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"✅ Datos cargados desde SQLite: {len(df)} partidos")
        return df

    def _cargar_datos_historicos_csv_fallback(self):
        """Método original de carga CSV (Backup). Años dinámicos: (año_actual - AÑOS_ELO_ATRAS) .. año_actual."""
        import os
        try:
            dfs = []
            current_year = date.today().year
            for año in range(current_year - AÑOS_ELO_ATRAS, current_year + 1):
                if año < 2018:
                    continue
                try:
                    # TML-Database usa 2022.csv, 2023.csv; nosotros también aceptamos atp_matches_2022_tml.csv
                    file_path = None
                    for p in (
                        os.path.join("datos", "raw", f"{año}.csv"),
                        os.path.join("datos", "raw", f"atp_matches_{año}_tml.csv"),
                    ):
                        if os.path.isfile(p):
                            file_path = p
                            break
                    if file_path is None:
                        continue
                    df_año = pd.read_csv(file_path)
                    df_año["tourney_date"] = pd.to_datetime(df_año["tourney_date"])
                    dfs.append(df_año)
                except Exception:
                    pass
            
            if not dfs:
                raise Exception("No CSVs found")
                
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values("tourney_date").reset_index(drop=True)
            logger.info(f"✅ Fallback CSV exitoso: {len(df)} partidos")
            return df
        except Exception as e:
             logger.error(f"❌ Fallback CSV falló: {e}")
             # Retornar DF vacío para no romper todo, aunque las predicciones sean malas
             return pd.DataFrame(columns=['tourney_date', 'winner_name', 'loser_name', 'surface'])

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
                r = ultimo.get("winner_rank", 999)
            else:
                r = ultimo.get("loser_rank", 999)
            # Evitar None/NaN (pandas puede devolverlos)
            if r is None or (isinstance(r, float) and pd.isna(r)):
                return 999
            return int(r) if r == r else 999  # NaN != NaN
        return 999

    def actualizar_con_partido(
        self,
        winner_name: str,
        loser_name: str,
        surface: str,
        winner_rank: int,
        loser_rank: int,
        fecha,
    ) -> None:
        """
        Actualiza el estado interno después de un partido completado (igual que backtesting).
        Añade el partido al histórico y actualiza ELO; los demás calculadores siguen
        usando el df inicial (como en backtesting).

        Args:
            winner_name: Nombre del ganador
            loser_name: Nombre del perdedor
            surface: Superficie (Hard/Clay/Grass)
            winner_rank: Ranking del ganador
            loser_rank: Ranking del perdedor
            fecha: Fecha del partido (date o datetime)
        """
        try:
            surface = self._normalizar_superficie(surface)
            fecha = pd.to_datetime(fecha)
            nuevo_partido = pd.DataFrame(
                [
                    {
                        "tourney_date": fecha,
                        "tourney_name": "",
                        "surface": surface,
                        "winner_name": winner_name,
                        "loser_name": loser_name,
                        "winner_rank": winner_rank if winner_rank else 999,
                        "loser_rank": loser_rank if loser_rank else 999,
                        "score": "",
                    }
                ]
            )
            self.df_historico = pd.concat(
                [self.df_historico, nuevo_partido], ignore_index=True
            )
            self.df_historico = self.df_historico.sort_values("tourney_date").reset_index(
                drop=True
            )
            self.elo_system.update_ratings(winner_name, loser_name, surface)
            logger.debug(
                f"📊 Feature state updated: {winner_name} d. {loser_name} ({surface})"
            )
        except Exception as e:
            logger.warning(f"⚠️  actualizar_con_partido failed: {e}")

    def _normalizar_superficie(self, superficie: str) -> str:
        """Mapea superficies a Hard/Clay/Grass (igual que backtesting)."""
        if not superficie:
            return "Hard"
        m = {"Outdoor": "Hard", "Indoor": "Hard", "Carpet": "Hard", "Hard": "Hard", "Clay": "Clay", "Grass": "Grass"}
        return m.get(str(superficie).strip(), "Hard")

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

        # 1. ELO (7 features) - defensivo ante None/NaN
        def _num(x, d=0):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return d
            return float(x)

        elo_j = _num(self.elo_system.get_rating(jugador_normalizado), 1500)
        elo_o = _num(self.elo_system.get_rating(oponente_normalizado), 1500)
        elo_j_surf = _num(self.elo_system.get_rating(jugador_normalizado, superficie), 1500)
        elo_o_surf = _num(self.elo_system.get_rating(oponente_normalizado, superficie), 1500)

        features["jugador_elo"] = elo_j
        features["oponente_elo"] = elo_o
        features["elo_diff"] = elo_j - elo_o
        features["jugador_elo_surface"] = elo_j_surf
        features["oponente_elo_surface"] = elo_o_surf
        features["elo_diff_surface"] = elo_j_surf - elo_o_surf
        # Mismo criterio que backtesting: probabilidad ELO por superficie (Hard/Clay/Grass)
        features["elo_expected_prob"] = self.elo_system.expected_score(elo_j_surf, elo_o_surf)
        
        logger.debug(f"  ELO: {jugador_normalizado}={elo_j:.0f} vs {oponente_normalizado}={elo_o:.0f} (diff={elo_j-elo_o:.0f})")

        # 2. Rankings (4 features)
        rank_j = self._obtener_ranking(jugador_normalizado, fecha)
        rank_o = self._obtener_ranking(oponente_normalizado, fecha)
        rank_j = _num(rank_j, 999)
        rank_o = _num(rank_o, 999)

        features["jugador_rank"] = rank_j
        features["oponente_rank"] = rank_o
        features["rank_diff"] = rank_o - rank_j
        features["rank_ratio"] = rank_j / max(rank_o, 1)

        # 3. Forma reciente (1 feature)
        forma_j = self.forma_calc.calcular_forma(jugador_normalizado, fecha, ventana_dias=60)
        forma_o = self.forma_calc.calcular_forma(oponente_normalizado, fecha, ventana_dias=60)
        wr_j = _num(forma_j.get("win_rate_60d"), 0.5)
        wr_o = _num(forma_o.get("win_rate_60d"), 0.5)
        features["diff_win_rate_60d"] = wr_j - wr_o

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
        features["serve_vs_return_advantage"] = _num(matchup.get("serve_vs_return_advantage"), 0)

        # 5. Superficie (1 feature) - pasar superficie normalizada (Hard/Clay/Grass)
        try:
            sup_norm = self._normalizar_superficie(superficie)
            ventaja_sup = self.superficie_calc.calcular_ventaja_superficie(
                jugador_normalizado, oponente_normalizado, fecha, sup_norm
            )
            features["ventaja_superficie"] = _num(ventaja_sup.get("ventaja_superficie"), 0)
        except Exception:
            features["ventaja_superficie"] = 0

        # 6. Interacciones (3 features)
        features["rank_diff_x_forma"] = features["rank_diff"] * wr_j
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
