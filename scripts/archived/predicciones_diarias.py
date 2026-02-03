"""
Script de Predicciones Diarias
================================

Script que se ejecuta automáticamente cada día (recomendado: 6 AM) para:
1. Obtener partidos del día desde TML Database
2. Generar predicciones automáticamente
3. Guardar en base de datos
4. Enviar alertas por email (opcional)

Uso:
    # Ejecución normal
    python scripts/predicciones_diarias.py

    # Dry run (no guarda en BD)
    python scripts/predicciones_diarias.py --dry-run

    # Especificar fecha
    python scripts/predicciones_diarias.py --date 2026-01-15

Cron job (ejecutar cada día a las 6 AM):
    0 6 * * * cd /path/to/tennis-ml-predictor && python scripts/predicciones_diarias.py
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date, timedelta
import logging
import pandas as pd

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.data.tml_data_downloader import TMLDataDownloader
from src.prediction.predictor_calibrado import PredictorCalibrado
from src.tracking.database_setup import TennisDatabase

# Configurar logging
log_file = Config.LOG_DIR / f"predicciones_diarias_{date.today()}.log"
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PredictorDiario:
    """Generador de predicciones diarias"""

    def __init__(self, modelo_path: str, db_path: str, dry_run: bool = False):
        """
        Args:
            modelo_path: Path al modelo entrenado
            db_path: Path a la base de datos
            dry_run: Si True, no guarda en base de datos
        """
        self.modelo_path = modelo_path
        self.db_path = db_path
        self.dry_run = dry_run

        # Inicializar componentes
        logger.info("Inicializando componentes...")
        self.predictor = PredictorCalibrado(modelo_path)
        self.db = TennisDatabase(db_path)
        self.downloader = TMLDataDownloader()

        logger.info("✅ Componentes inicializados")

    def obtener_partidos_del_dia(self, fecha: date = None) -> pd.DataFrame:
        """
        Obtiene partidos del día desde TML Database

        Args:
            fecha: Fecha a consultar (default: hoy)

        Returns:
            DataFrame con partidos del día
        """
        if fecha is None:
            fecha = date.today()

        logger.info(f"📅 Obteniendo partidos para {fecha}")

        try:
            # Descargar datos recientes (últimos 7 días para contexto)
            fecha_inicio = fecha - timedelta(days=7)
            fecha_fin = fecha + timedelta(days=1)

            df = self.downloader.download_matches(
                start_date=fecha_inicio.strftime("%Y-%m-%d"),
                end_date=fecha_fin.strftime("%Y-%m-%d"),
            )

            if df.empty:
                logger.warning(f"⚠️  No se encontraron partidos para {fecha}")
                return pd.DataFrame()

            # Filtrar solo partidos del día especificado
            df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
            df_dia = df[df["match_date"] == fecha].copy()

            logger.info(f"✅ Encontrados {len(df_dia)} partidos para {fecha}")
            return df_dia

        except Exception as e:
            logger.error(f"❌ Error obteniendo partidos: {e}", exc_info=True)
            return pd.DataFrame()

    def generar_predicciones(self, df_partidos: pd.DataFrame) -> list:
        """
        Genera predicciones para los partidos

        Args:
            df_partidos: DataFrame con partidos

        Returns:
            Lista de predicciones
        """
        if df_partidos.empty:
            return []

        predicciones = []

        logger.info(f"🤖 Generando predicciones para {len(df_partidos)} partidos...")

        for idx, partido in df_partidos.iterrows():
            try:
                # Extraer información del partido
                jugador1 = partido.get("player1_name", "")
                jugador2 = partido.get("player2_name", "")
                jugador1_rank = partido.get("player1_rank", 100)
                jugador2_rank = partido.get("player2_rank", 100)
                superficie = partido.get("surface", "Hard")

                # Cuota por defecto (se puede mejorar integrando con bookmakers)
                # Por ahora usamos una cuota estimada basada en rankings
                cuota_estimada = self._estimar_cuota(jugador1_rank, jugador2_rank)

                # Generar predicción
                resultado = self.predictor.predecir_partido(
                    jugador1=jugador1,
                    jugador1_rank=jugador1_rank,
                    jugador2=jugador2,
                    jugador2_rank=jugador2_rank,
                    superficie=superficie,
                    cuota=cuota_estimada,
                )

                # Preparar datos para guardar
                prediccion = {
                    "fecha_partido": partido.get("match_date"),
                    "jugador1": jugador1,
                    "jugador2": jugador2,
                    "superficie": superficie,
                    "probabilidad": resultado.get("probabilidad", 0.5),
                    "cuota": cuota_estimada,
                    "expected_value": resultado.get("expected_value", 0),
                    "decision": (
                        "APOSTAR ✅"
                        if resultado.get("expected_value", 0) > Config.EV_THRESHOLD
                        else "NO APOSTAR ❌"
                    ),
                    "fecha_prediccion": datetime.now(),
                }

                predicciones.append(prediccion)

                # Log
                ev = resultado.get("expected_value", 0)
                logger.info(
                    f"  {jugador1} vs {jugador2}: P={resultado.get('probabilidad', 0):.2%}, EV={ev:.2%}"
                )

            except Exception as e:
                logger.error(f"❌ Error prediciendo {jugador1} vs {jugador2}: {e}")
                continue

        logger.info(f"✅ Generadas {len(predicciones)} predicciones")
        return predicciones

    def _estimar_cuota(self, rank1: int, rank2: int) -> float:
        """
        Estima una cuota basada en los rankings

        Args:
            rank1: Ranking jugador 1
            rank2: Ranking jugador 2

        Returns:
            Cuota estimada
        """
        # Diferencia de ranking
        diff = rank2 - rank1

        # Cuota estimada (muy simplificado)
        if diff > 50:
            return 1.30
        elif diff > 20:
            return 1.50
        elif diff > 0:
            return 1.70
        elif diff > -20:
            return 2.00
        elif diff > -50:
            return 2.50
        else:
            return 3.00

    def guardar_predicciones(self, predicciones: list) -> int:
        """
        Guarda predicciones en la base de datos

        Args:
            predicciones: Lista de predicciones

        Returns:
            Número de predicciones guardadas
        """
        if not predicciones:
            logger.warning("⚠️  No hay predicciones para guardar")
            return 0

        if self.dry_run:
            logger.info("🔍 DRY RUN: No se guardan predicciones en BD")
            return len(predicciones)

        guardadas = 0

        for pred in predicciones:
            try:
                # Guardar en base de datos
                self.db.registrar_prediccion(
                    jugador1=pred["jugador1"],
                    jugador2=pred["jugador2"],
                    superficie=pred["superficie"],
                    probabilidad=pred["probabilidad"],
                    cuota=pred["cuota"],
                    decision=pred["decision"],
                    fecha_partido=pred["fecha_partido"],
                )
                guardadas += 1

            except Exception as e:
                logger.error(f"❌ Error guardando predicción: {e}")
                continue

        logger.info(f"✅ Guardadas {guardadas}/{len(predicciones)} predicciones en BD")
        return guardadas

    def enviar_alertas(self, predicciones: list):
        """
        Envía alertas por email con predicciones de alto valor

        Args:
            predicciones: Lista de predicciones
        """
        if not Config.EMAIL_ENABLED:
            logger.info("📧 Email deshabilitado, no se envían alertas")
            return

        # Filtrar predicciones con EV alto
        predicciones_altas = [
            p for p in predicciones if p["expected_value"] > Config.EV_THRESHOLD_ALERT
        ]

        if not predicciones_altas:
            logger.info("📧 No hay predicciones de alto valor para alertar")
            return

        try:
            from src.bookmakers.alert_system import AlertSystem

            alert_system = AlertSystem()

            # Preparar mensaje
            mensaje = f"🎾 Predicciones de Alto Valor - {date.today()}\n\n"

            for pred in predicciones_altas:
                mensaje += f"🔥 {pred['jugador1']} vs {pred['jugador2']}\n"
                mensaje += f"   Probabilidad: {pred['probabilidad']:.2%}\n"
                mensaje += f"   Cuota: {pred['cuota']:.2f}\n"
                mensaje += f"   EV: {pred['expected_value']:.2%}\n"
                mensaje += f"   Superficie: {pred['superficie']}\n\n"

            alert_system.enviar_email(
                asunto=f"🎾 {len(predicciones_altas)} Predicciones de Alto Valor", mensaje=mensaje
            )

            logger.info(f"✅ Email enviado con {len(predicciones_altas)} predicciones")

        except Exception as e:
            logger.error(f"❌ Error enviando email: {e}")

    def ejecutar(self, fecha: date = None):
        """
        Ejecuta el proceso completo de predicciones diarias

        Args:
            fecha: Fecha a procesar (default: hoy)
        """
        logger.info("=" * 70)
        logger.info("🎾 PREDICCIONES DIARIAS - INICIO")
        logger.info("=" * 70)

        if fecha is None:
            fecha = date.today()

        logger.info(f"📅 Fecha: {fecha}")
        logger.info(f"🔍 Dry Run: {self.dry_run}")

        # 1. Obtener partidos del día
        df_partidos = self.obtener_partidos_del_dia(fecha)

        if df_partidos.empty:
            logger.warning("⚠️  No hay partidos para procesar")
            return

        # 2. Generar predicciones
        predicciones = self.generar_predicciones(df_partidos)

        if not predicciones:
            logger.warning("⚠️  No se generaron predicciones")
            return

        # 3. Guardar en base de datos
        guardadas = self.guardar_predicciones(predicciones)

        # 4. Enviar alertas
        self.enviar_alertas(predicciones)

        # Resumen
        logger.info("=" * 70)
        logger.info("📊 RESUMEN")
        logger.info("=" * 70)
        logger.info(f"  Partidos encontrados: {len(df_partidos)}")
        logger.info(f"  Predicciones generadas: {len(predicciones)}")
        logger.info(f"  Predicciones guardadas: {guardadas}")

        apuestas_recomendadas = sum(1 for p in predicciones if "APOSTAR" in p["decision"])
        logger.info(f"  Apuestas recomendadas: {apuestas_recomendadas}")

        logger.info("=" * 70)
        logger.info("✅ PREDICCIONES DIARIAS - COMPLETADO")
        logger.info("=" * 70)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Generador de predicciones diarias")
    parser.add_argument(
        "--dry-run", action="store_true", help="No guardar en base de datos (solo simular)"
    )
    parser.add_argument("--date", type=str, help="Fecha a procesar (formato: YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default=Config.MODEL_PATH, help="Path al modelo")
    parser.add_argument("--db", type=str, default=Config.DB_PATH, help="Path a la base de datos")

    args = parser.parse_args()

    # Parsear fecha
    fecha = None
    if args.date:
        try:
            fecha = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"❌ Formato de fecha inválido: {args.date}")
            return

    # Ejecutar
    try:
        predictor = PredictorDiario(modelo_path=args.model, db_path=args.db, dry_run=args.dry_run)

        predictor.ejecutar(fecha)

    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
