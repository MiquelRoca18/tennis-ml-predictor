"""
Backtesting de Producción REAL - Versión Completa
==================================================

Simula EXACTAMENTE cómo funcionaría el modelo en producción:
1. Genera las 30 features completas usando los calculadores reales
2. Usa predicción bidireccional (Opción B): predice ambas direcciones y promedia
3. Evalúa con cuotas reales de 2024
4. Aplica Kelly Criterion para gestión de bankroll

Este script replica fielmente el proceso de producción.
"""

import pandas as pd
import joblib
from pathlib import Path
import logging
import sys
import requests
import subprocess

# Añadir path para imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.features.elo_rating_system import TennisELO
from src.features.features_servicio_resto import ServicioRestoCalculator
from src.features.features_fatiga import FatigaCalculator
from src.features.features_forma_reciente import FormaRecienteCalculator
from src.features.features_h2h_mejorado import HeadToHeadCalculator
from src.features.features_superficie import SuperficieSpecializationCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def descargar_datos_automaticamente(año):
    """
    Descarga automáticamente datos de partidos y cuotas para el año especificado

    Args:
        año: Año para el cual descargar datos (ej: 2024, 2025)

    Returns:
        tuple: (ruta_partidos, ruta_cuotas)
    """
    logger.info(f"\n📥 Descargando datos para {año}...")

    datos_dir = Path("datos/raw")
    odds_dir = Path("datos/odds_historicas")
    datos_dir.mkdir(parents=True, exist_ok=True)
    odds_dir.mkdir(parents=True, exist_ok=True)

    # Rutas de archivos
    partidos_file = datos_dir / f"atp_matches_{año}_tml.csv"
    cuotas_file = odds_dir / f"tennis_odds_{año}_{año}.csv"
    cuotas_excel = odds_dir / f"{año}.xlsx"

    # 1. Descargar datos de partidos (TML GitHub)
    if not partidos_file.exists():
        logger.info(f"  Descargando partidos {año} de TML...")
        url_partidos = (
            f"https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{año}.csv"
        )
        try:
            response = requests.get(url_partidos, timeout=30)
            response.raise_for_status()
            partidos_file.write_bytes(response.content)
            logger.info(f"  ✅ Partidos {año} descargados: {partidos_file}")
        except Exception as e:
            logger.error(f"  ❌ Error descargando partidos: {e}")
            raise
    else:
        logger.info(f"  ✅ Partidos {año} ya existen: {partidos_file}")

    # 2. Descargar cuotas (Tennis-Data.co.uk)
    if not cuotas_file.exists():
        logger.info(f"  Descargando cuotas {año} de Tennis-Data.co.uk...")
        url_cuotas = f"http://www.tennis-data.co.uk/{año}/{año}.xlsx"
        try:
            response = requests.get(url_cuotas, timeout=30)
            response.raise_for_status()
            cuotas_excel.write_bytes(response.content)
            logger.info(f"  ✅ Cuotas Excel descargadas: {cuotas_excel}")

            # Procesar Excel a CSV
            logger.info(f"  Procesando cuotas...")
            result = subprocess.run(
                ["python", "scripts/internal/process_tennis_data_odds.py"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Separar solo el año específico
                df_all = pd.read_csv(odds_dir / f"tennis_odds_2024_{año}.csv")
                df_all["fecha"] = pd.to_datetime(df_all["fecha"])
                df_año = df_all[df_all["fecha"].dt.year == año].copy()
                df_año.to_csv(cuotas_file, index=False)
                logger.info(f"  ✅ Cuotas {año} procesadas: {len(df_año)} partidos")
            else:
                logger.error(f"  ❌ Error procesando cuotas: {result.stderr}")
                raise Exception("Error procesando cuotas")

        except Exception as e:
            logger.error(f"  ❌ Error descargando cuotas: {e}")
            raise
    else:
        logger.info(f"  ✅ Cuotas {año} ya existen: {cuotas_file}")

    logger.info(f"\n✅ Datos {año} listos")
    return str(partidos_file), str(cuotas_file)


class ProductionFeatureGenerator:
    """
    Generador de features que replica EXACTAMENTE el proceso de entrenamiento
    """

    def __init__(self, df_historico):
        """
        Inicializa todos los calculadores con el histórico disponible
        """
        logger.info("📊 Inicializando calculadores de features...")

        self.df = df_historico.copy()
        self.df["tourney_date"] = pd.to_datetime(self.df["tourney_date"])

        # Inicializar ELO
        logger.info("  - Sistema ELO...")
        self.elo_system = TennisELO(k_factor=32, base_rating=1500)
        self.df = self.elo_system.calculate_historical_elos(self.df)

        # Otros calculadores
        logger.info("  - Servicio/Resto...")
        self.servicio_calc = ServicioRestoCalculator(self.df)

        logger.info("  - Fatiga...")
        self.fatiga_calc = FatigaCalculator(self.df)

        logger.info("  - Forma reciente...")
        self.forma_calc = FormaRecienteCalculator(self.df)

        logger.info("  - Head-to-Head...")
        self.h2h_calc = HeadToHeadCalculator(self.df)

        logger.info("  - Superficie...")
        self.superficie_calc = SuperficieSpecializationCalculator(self.df)

        logger.info("✅ Calculadores inicializados")

    def actualizar_con_partido(
        self, winner_name, loser_name, surface, winner_rank, loser_rank, fecha
    ):
        """
        Actualiza los calculadores después de un partido
        """
        # Crear fila del partido
        nuevo_partido = pd.DataFrame(
            [
                {
                    "tourney_date": fecha,
                    "winner_name": winner_name,
                    "loser_name": loser_name,
                    "surface": surface,
                    "winner_rank": winner_rank,
                    "loser_rank": loser_rank,
                }
            ]
        )

        # Añadir al histórico
        self.df = pd.concat([self.df, nuevo_partido], ignore_index=True)
        self.df = self.df.sort_values("tourney_date").reset_index(drop=True)

        # Actualizar ELO
        self.elo_system.update_ratings(winner_name, loser_name, surface)

        # Los otros calculadores se actualizan automáticamente al consultar self.df

    def generar_features(self, jugador, oponente, superficie, fecha):
        """
        Genera las 30 features para un partido

        IMPORTANTE: jugador está en posición 'jugador', oponente en posición 'oponente'
        El modelo predice: ¿Ganará 'jugador'?
        """
        features = {}

        # 1. ELO (7 features)
        elo_j = self.elo_system.get_rating(jugador)
        elo_o = self.elo_system.get_rating(oponente)
        elo_j_surf = self.elo_system.get_rating(jugador, superficie)
        elo_o_surf = self.elo_system.get_rating(oponente, superficie)

        features["jugador_elo"] = elo_j
        features["oponente_elo"] = elo_o
        features["elo_diff"] = elo_j - elo_o
        features["jugador_elo_surface"] = elo_j_surf
        features["oponente_elo_surface"] = elo_o_surf
        features["elo_diff_surface"] = elo_j_surf - elo_o_surf
        features["elo_expected_prob"] = self.elo_system.expected_score(elo_j, elo_o)

        # 2. Rankings (4 features)
        rank_j = self._obtener_ranking(jugador, fecha)
        rank_o = self._obtener_ranking(oponente, fecha)

        features["jugador_rank"] = rank_j
        features["oponente_rank"] = rank_o
        features["rank_diff"] = rank_o - rank_j
        features["rank_ratio"] = rank_j / max(rank_o, 1)

        # 3. Forma reciente (1 feature)
        forma_j = self.forma_calc.calcular_forma(jugador, fecha, ventana_dias=60)
        forma_o = self.forma_calc.calcular_forma(oponente, fecha, ventana_dias=60)

        features["diff_win_rate_60d"] = forma_j.get("win_rate_60d", 0.5) - forma_o.get(
            "win_rate_60d", 0.5
        )

        # 4. Servicio y Resto (14 features)
        serv_j = self.servicio_calc.calcular_estadisticas_servicio(jugador, fecha)
        serv_o = self.servicio_calc.calcular_estadisticas_servicio(oponente, fecha)
        resto_j = self.servicio_calc.calcular_estadisticas_resto(jugador, fecha)

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

        matchup = self.servicio_calc.calcular_matchup_servicio_resto(jugador, oponente, fecha)
        features["serve_vs_return_advantage"] = matchup.get("serve_vs_return_advantage", 0)

        # 5. Superficie (1 feature)
        try:
            ventaja_sup = self.superficie_calc.calcular_ventaja_superficie(
                jugador, oponente, fecha, superficie
            )
            features["ventaja_superficie"] = ventaja_sup.get("ventaja_superficie", 0)
        except Exception:
            # Si falla el cálculo de superficie, usar valor neutral
            features["ventaja_superficie"] = 0

        # 6. Interacciones (3 features)
        features["rank_diff_x_forma"] = features["rank_diff"] * forma_j.get("win_rate_60d", 0.5)
        features["elo_x_forma"] = features["elo_diff"] * features["diff_win_rate_60d"]
        features["superficie_x_rank"] = features["ventaja_superficie"] * features["rank_diff"]

        return features

    def _obtener_ranking(self, jugador, fecha):
        """Obtiene el ranking más reciente de un jugador antes de la fecha"""
        partidos = self.df[
            ((self.df["winner_name"] == jugador) | (self.df["loser_name"] == jugador))
            & (self.df["tourney_date"] < fecha)
        ].sort_values("tourney_date", ascending=False)

        if len(partidos) > 0:
            ultimo = partidos.iloc[0]
            if ultimo["winner_name"] == jugador:
                return ultimo.get("winner_rank", 999)
            else:
                return ultimo.get("loser_rank", 999)
        return 999


class BacktestingProduccionReal:
    """
    Backtesting que simula EXACTAMENTE el proceso de producción
    """

    def __init__(
        self,
        modelo_path,
        bankroll_inicial=1000.0,
        kelly_fraction=0.05,
        umbral_ev=0.10,
        max_cuota=2.0,
        min_probabilidad=0.60,
    ):
        self.modelo_path = Path(modelo_path)
        self.bankroll_inicial = bankroll_inicial
        self.kelly_fraction = kelly_fraction
        self.umbral_ev = umbral_ev
        self.max_cuota = max_cuota  # Solo favoritos
        self.min_probabilidad = min_probabilidad  # Solo cuando modelo está seguro
        self.resultados_dir = Path("resultados/backtesting_produccion_real")
        self.resultados_dir.mkdir(parents=True, exist_ok=True)

        self.modelo = None
        self.feature_cols = None

        logger.info(f"🎯 Backtesting de Producción REAL - ESTRATEGIA CONSERVADORA")
        logger.info(f"💰 Bankroll inicial: {bankroll_inicial}€")
        logger.info(f"📊 Kelly Fraction: {kelly_fraction*100:.1f}% (MUY CONSERVADOR)")
        logger.info(f"📈 Umbral EV mínimo: {umbral_ev*100:.0f}%")
        logger.info(f"🎲 Cuota máxima: {max_cuota} (solo favoritos)")
        logger.info(f"🎯 Probabilidad mínima: {min_probabilidad*100:.0f}%")

    def cargar_modelo(self):
        """Carga el modelo y las features"""
        logger.info(f"\n📂 Cargando modelo...")
        self.modelo = joblib.load(self.modelo_path)

        # Cargar features
        with open("resultados/selected_features.txt", "r") as f:
            self.feature_cols = [line.strip() for line in f.readlines() if line.strip()]

        logger.info(f"✅ Modelo cargado ({len(self.feature_cols)} features)")

    def normalizar_nombre(self, nombre):
        """Normaliza nombre"""
        if pd.isna(nombre):
            return ""
        nombre = str(nombre).lower().strip()
        replacements = {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ñ": "n",
            "ü": "u",
            "ç": "c",
            "-": " ",
            ".": "",
            "'": "",
        }
        for old, new in replacements.items():
            nombre = nombre.replace(old, new)
        return " ".join(nombre.split())

    def buscar_jugador_nombre_completo(self, nombre_odds, df_historico):
        """Busca el nombre completo de un jugador"""
        apellido_odds, inicial_odds = self._extraer_apellido_inicial(nombre_odds)

        for nombre in df_historico["winner_name"].unique():
            apellido, inicial = self._extraer_apellido_inicial(nombre)
            if apellido == apellido_odds and inicial == inicial_odds:
                return nombre

        for nombre in df_historico["loser_name"].unique():
            apellido, inicial = self._extraer_apellido_inicial(nombre)
            if apellido == apellido_odds and inicial == inicial_odds:
                return nombre

        return None

    def _extraer_apellido_inicial(self, nombre_completo):
        """Extrae apellido e inicial"""
        if pd.isna(nombre_completo):
            return ("", "")

        nombre_norm = self.normalizar_nombre(nombre_completo)
        partes = nombre_norm.split()

        if len(partes) == 0:
            return ("", "")
        elif len(partes) == 1:
            return (partes[0], "")
        elif len(partes) == 2:
            if len(partes[1]) == 1:
                return (partes[0], partes[1])
            else:
                return (partes[1], partes[0][0] if partes[0] else "")
        else:
            return (partes[-1], partes[0][0] if partes[0] else "")

    def predecir_partido_bidireccional(self, feature_gen, jugador1, jugador2, superficie, fecha):
        """
        Predicción con NUEVO MODELO (formato j1_/j2_):
        - Genera features para jugador1 (prefijo j1_)
        - Genera features para jugador2 (prefijo j2_)
        - Hace 1 predicción: ¿Ganará j1?

        Returns:
            prob_jugador1_gana: Probabilidad de que jugador1 gane
        """
        # Generar features para jugador1 (como 'jugador')
        features_j1 = feature_gen.generar_features(jugador1, jugador2, superficie, fecha)

        # Generar features para jugador2 (como 'jugador')
        features_j2 = feature_gen.generar_features(jugador2, jugador1, superficie, fecha)

        # Combinar con prefijos j1_ y j2_
        features_combined = {}
        for key, value in features_j1.items():
            features_combined[f"j1_{key}"] = value
        for key, value in features_j2.items():
            features_combined[f"j2_{key}"] = value

        # Preparar y predecir
        X = self._preparar_features(features_combined)
        prob_j1_gana = self.modelo.predict_proba(X)[0, 1]  # P(j1 gana)

        return prob_j1_gana

    def _preparar_features(self, features_dict):
        """Prepara features en el orden correcto para el modelo"""
        features_array = []
        for feat_name in self.feature_cols:
            if feat_name in features_dict:
                features_array.append(features_dict[feat_name])
            else:
                features_array.append(0.0)

        return pd.DataFrame([features_array], columns=self.feature_cols)

    def calcular_kelly_stake(self, prob_modelo, cuota, bankroll_actual):
        """Calcula stake usando Kelly Criterion"""
        prob_implicita = 1.0 / cuota

        if prob_modelo <= prob_implicita:
            return 0.0

        kelly_pct = (prob_modelo * cuota - 1) / (cuota - 1)
        kelly_pct = kelly_pct * self.kelly_fraction
        kelly_pct = min(kelly_pct, 0.10)

        if kelly_pct <= 0.01:
            return 0.0

        stake = bankroll_actual * kelly_pct

        if stake < 5.0:
            return 0.0

        return stake

    def calcular_ev(self, prob, cuota):
        """Calcula Expected Value"""
        return (prob * cuota) - 1

    def ejecutar_backtesting(self, df_odds, df_historico):
        """Ejecuta backtesting completo"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🎲 BACKTESTING DE PRODUCCIÓN REAL - AÑO 2024")
        logger.info(f"{'='*70}")
        logger.info(f"Total partidos con cuotas: {len(df_odds)}")

        # Inicializar generador de features con histórico hasta 2024
        df_historico_pre2024 = df_historico[df_historico["tourney_date"] < "2024-01-01"].copy()
        feature_gen = ProductionFeatureGenerator(df_historico_pre2024)

        # Ordenar partidos cronológicamente
        df_odds = df_odds.sort_values("fecha").reset_index(drop=True)

        bankroll_actual = self.bankroll_inicial
        apuestas_realizadas = []
        bankroll_history = [self.bankroll_inicial]
        partidos_sin_jugadores = 0
        partidos_procesados = 0

        logger.info(f"\n🔄 Procesando partidos...")
        logger.info(f"{'='*70}\n")

        for idx, partido_odds in df_odds.iterrows():
            # 1. BUSCAR NOMBRES COMPLETOS
            j1_nombre = self.buscar_jugador_nombre_completo(partido_odds["jugador_1"], df_historico)
            j2_nombre = self.buscar_jugador_nombre_completo(partido_odds["jugador_2"], df_historico)

            if not j1_nombre or not j2_nombre:
                partidos_sin_jugadores += 1
                continue

            # Normalizar superficie
            superficie = partido_odds.get("superficie", "Hard")
            # Mapear superficies no estándar a las 3 principales
            superficie_map = {
                "Outdoor": "Hard",
                "Indoor": "Hard",
                "Carpet": "Hard",
                "Hard": "Hard",
                "Clay": "Clay",
                "Grass": "Grass",
            }
            superficie = superficie_map.get(superficie, "Hard")

            fecha = pd.to_datetime(partido_odds["fecha"])

            # 2. PREDICCIÓN BIDIRECCIONAL (Opción B)
            try:
                prob_j1_gana = self.predecir_partido_bidireccional(
                    feature_gen, j1_nombre, j2_nombre, superficie, fecha
                )
            except Exception as e:
                logger.warning(f"Error en predicción para {j1_nombre} vs {j2_nombre}: {str(e)}")
                # Actualizar feature generator incluso si falla la predicción
                feature_gen.actualizar_con_partido(
                    j1_nombre,
                    j2_nombre,
                    superficie,
                    partido_odds.get("ganador_rank", 999),
                    partido_odds.get("perdedor_rank", 999),
                    fecha,
                )
                continue

            partidos_procesados += 1

            # 3. DECIDIR APUESTA (ESTRATEGIA CONSERVADORA)
            cuota_j1 = partido_odds["cuota_jugador_1"]
            cuota_j2 = partido_odds["cuota_jugador_2"]

            ev_j1 = self.calcular_ev(prob_j1_gana, cuota_j1)
            ev_j2 = self.calcular_ev(1 - prob_j1_gana, cuota_j2)

            # FILTROS CONSERVADORES:
            # 1. Solo favoritos (cuota < max_cuota)
            # 2. Solo cuando modelo está seguro (prob > min_probabilidad)
            # 3. EV debe superar umbral alto

            # Apostar al jugador con mayor EV (si supera todos los filtros)
            if (
                ev_j1 > self.umbral_ev
                and ev_j1 > ev_j2
                and cuota_j1 < self.max_cuota
                and prob_j1_gana > self.min_probabilidad
            ):
                # Apostar a jugador 1
                stake = self.calcular_kelly_stake(prob_j1_gana, cuota_j1, bankroll_actual)
                if stake > 0:
                    # jugador_1 SIEMPRE es el ganador en los datos
                    ganancia = stake * (cuota_j1 - 1)
                    bankroll_actual += ganancia

                    apuestas_realizadas.append(
                        {
                            "fecha": fecha,
                            "partido_num": len(apuestas_realizadas) + 1,
                            "jugador_apostado": j1_nombre,
                            "oponente": j2_nombre,
                            "prob_modelo": prob_j1_gana,
                            "cuota": cuota_j1,
                            "ev": ev_j1,
                            "stake": stake,
                            "resultado": 1,
                            "ganancia": ganancia,
                            "bankroll_despues": bankroll_actual,
                        }
                    )
                    bankroll_history.append(bankroll_actual)

            elif (
                ev_j2 > self.umbral_ev
                and cuota_j2 < self.max_cuota
                and (1 - prob_j1_gana) > self.min_probabilidad
            ):
                # Apostar a jugador 2
                stake = self.calcular_kelly_stake(1 - prob_j1_gana, cuota_j2, bankroll_actual)
                if stake > 0:
                    # jugador_2 SIEMPRE es el perdedor en los datos
                    ganancia = -stake
                    bankroll_actual += ganancia

                    apuestas_realizadas.append(
                        {
                            "fecha": fecha,
                            "partido_num": len(apuestas_realizadas) + 1,
                            "jugador_apostado": j2_nombre,
                            "oponente": j1_nombre,
                            "prob_modelo": 1 - prob_j1_gana,
                            "cuota": cuota_j2,
                            "ev": ev_j2,
                            "stake": stake,
                            "resultado": 0,
                            "ganancia": ganancia,
                            "bankroll_despues": bankroll_actual,
                        }
                    )
                    bankroll_history.append(bankroll_actual)

            # 4. ACTUALIZAR FEATURE GENERATOR con el resultado real
            feature_gen.actualizar_con_partido(
                j1_nombre,
                j2_nombre,
                superficie,
                partido_odds.get("ganador_rank", 999),
                partido_odds.get("perdedor_rank", 999),
                fecha,
            )

            # Log progreso
            if len(apuestas_realizadas) > 0 and len(apuestas_realizadas) % 50 == 0:
                logger.info(f"Apuestas realizadas: {len(apuestas_realizadas)}")
                logger.info(
                    f"  Bankroll: {bankroll_actual:.2f}€ ({(bankroll_actual/self.bankroll_inicial-1)*100:+.1f}%)\n"
                )

        logger.info(f"\n{'='*70}")
        logger.info(f"✅ BACKTESTING COMPLETADO")
        logger.info(f"{'='*70}")
        logger.info(f"Partidos con cuotas: {len(df_odds)}")
        logger.info(f"Partidos sin jugadores: {partidos_sin_jugadores}")
        logger.info(f"Partidos procesados: {partidos_procesados}")
        logger.info(f"Apuestas realizadas: {len(apuestas_realizadas)}")

        if len(apuestas_realizadas) > 0:
            df_apuestas = pd.DataFrame(apuestas_realizadas)
            self.mostrar_resultados(df_apuestas, bankroll_history)
            self.guardar_resultados(df_apuestas, bankroll_history)
        else:
            logger.error("No se realizaron apuestas")

        return apuestas_realizadas, bankroll_history

    def mostrar_resultados(self, df_apuestas, bankroll_history):
        """Muestra resultados"""
        total_apuestas = len(df_apuestas)
        ganadas = (df_apuestas["resultado"] == 1).sum()
        win_rate = ganadas / total_apuestas * 100

        bankroll_final = bankroll_history[-1]
        ganancia_neta = bankroll_final - self.bankroll_inicial
        roi = (ganancia_neta / self.bankroll_inicial) * 100

        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RESULTADOS FINALES")
        logger.info(f"{'='*70}")
        logger.info(f"\n💰 BANKROLL:")
        logger.info(f"  Inicial:  {self.bankroll_inicial:.2f}€")
        logger.info(f"  Final:    {bankroll_final:.2f}€")
        logger.info(f"  Ganancia: {ganancia_neta:+.2f}€")
        logger.info(f"  ROI:      {roi:+.2f}%")
        logger.info(f"\n🎯 PERFORMANCE:")
        logger.info(f"  Total:    {total_apuestas}")
        logger.info(f"  Ganadas:  {ganadas} ({win_rate:.1f}%)")
        logger.info(f"  Perdidas: {total_apuestas - ganadas} ({100-win_rate:.1f}%)")
        logger.info(f"{'='*70}")

    def guardar_resultados(self, df_apuestas, bankroll_history):
        """Guarda resultados"""
        df_apuestas.to_csv(self.resultados_dir / "apuestas_detalladas.csv", index=False)

        pd.DataFrame(
            {"apuesta_num": range(len(bankroll_history)), "bankroll": bankroll_history}
        ).to_csv(self.resultados_dir / "bankroll_evolution.csv", index=False)

        logger.info(f"\n💾 Resultados guardados en: {self.resultados_dir}")


def main():
    """
    Función principal
    """
    # CONFIGURACIÓN: Cambiar año aquí para backtesting de diferentes años
    AÑO_BACKTESTING = 2024  # Cambiar a 2024, 2025, etc.

    logger.info("\n" + "=" * 70)
    logger.info(f"🎾 BACKTESTING DE PRODUCCIÓN REAL - AÑO {AÑO_BACKTESTING}")
    logger.info("=" * 70)

    # Descargar datos automáticamente si no existen (usa TML)
    try:
        historico_file, odds_file = descargar_datos_automaticamente(AÑO_BACKTESTING)
    except Exception as e:
        logger.error(f"❌ Error descargando datos: {e}")
        logger.info(
            "\n💡 Solución: Verifica tu conexión a internet y que los datos estén disponibles"
        )
        return

    modelo_path = "modelos/random_forest_calibrado.pkl"

    # ESTRATEGIA CONSERVADORA (RECOMENDADA)
    backtester = BacktestingProduccionReal(
        modelo_path=modelo_path,
        bankroll_inicial=1000.0,
        kelly_fraction=0.05,  # 5% Kelly (muy conservador)
        umbral_ev=0.10,  # EV mínimo 10% (solo mejores oportunidades)
        max_cuota=2.0,  # Solo favoritos
        min_probabilidad=0.60,  # Solo cuando modelo está seguro (60%+)
    )

    backtester.cargar_modelo()

    logger.info(f"\n📂 Cargando datos...")
    df_odds = pd.read_csv(odds_file)
    df_odds["fecha"] = pd.to_datetime(df_odds["fecha"])
    logger.info(f"✅ {len(df_odds)} partidos con cuotas")

    df_historico = pd.read_csv(historico_file)
    df_historico["tourney_date"] = pd.to_datetime(df_historico["tourney_date"])
    logger.info(f"✅ {len(df_historico)} partidos históricos")

    apuestas, bankroll_history = backtester.ejecutar_backtesting(df_odds, df_historico)

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ PROCESO COMPLETADO")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
