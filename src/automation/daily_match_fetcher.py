"""
Daily Match Fetcher Service
============================

Fetches upcoming tennis matches from the Tennis API, creates them in the database,
and generates predictions automatically. Designed to run once per day to minimize
API calls while keeping data fresh.
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.api_tennis_client import APITennisClient
from src.database.match_database import MatchDatabase
from src.prediction.predictor_calibrado import PredictorCalibrado
from src.utils.tournament_surface_mapper import TournamentSurfaceMapper
from src.config.settings import Config

logger = logging.getLogger(__name__)


class DailyMatchFetcher:
    """
    Service to fetch daily matches and generate predictions
    """

    def __init__(
        self,
        db: MatchDatabase,
        api_client: APITennisClient,
        predictor: Optional[PredictorCalibrado] = None,
    ):
        """
        Initialize the daily match fetcher

        Args:
            db: Database instance
            api_client: Tennis API client
            predictor: ML predictor instance (opcional - sin él solo se guardan partidos sin predicciones)
        """
        self.db = db
        self.api_client = api_client
        self.predictor = predictor
        self.surface_mapper = TournamentSurfaceMapper()

        if predictor:
            logger.info("✅ DailyMatchFetcher initialized (con predictor)")
        else:
            logger.warning("⚠️  DailyMatchFetcher initialized SIN predictor - partidos se guardarán sin predicciones")

    def fetch_and_store_matches(self, days_ahead: int = 7) -> Dict:
        """
        Fetches upcoming matches from Tennis API and stores them with predictions

        This is the main method that should be called daily. It:
        1. Fetches matches for the next N days from Tennis API
        2. Checks if each match already exists in database
        3. Creates new matches with odds
        4. Generates predictions for all new matches
        5. Returns statistics about the operation

        Args:
            days_ahead: Number of days to fetch ahead (default: 7)

        Returns:
            Dict with statistics (matches_found, matches_new, matches_existing, api_calls, etc.)
        """
        logger.info(f"🔄 Starting daily match fetch (next {days_ahead} days)...")

        stats = {
            "timestamp": datetime.now().isoformat(),
            "days_ahead": days_ahead,
            "matches_found": 0,
            "matches_new": 0,
            "matches_existing": 0,
            "matches_filtered": 0,  # Partidos filtrados por tipo (WTA, dobles, etc.)
            "matches_created": [],
            "predictions_generated": 0,
            "api_calls_made": 0,
            "errors": [],
        }

        try:
            # 1. Fetch matches from Tennis API
            logger.info("📥 Fetching matches from Tennis API...")
            matches_raw = self.api_client.get_all_matches_with_odds(days_ahead=days_ahead)

            stats["matches_found"] = len(matches_raw)
            stats["api_calls_made"] = self.api_client.requests_made

            if not matches_raw:
                logger.warning("⚠️  No matches found from Tennis API")
                return stats

            logger.info(f"✅ Found {len(matches_raw)} matches from API")
            
            # DEBUG: Log first match structure to see available fields
            if matches_raw:
                first_match = matches_raw[0]
                logger.info(f"🔍 DEBUG - First match fields: event_type={first_match.get('event_type')}, league={first_match.get('league')}, player1={first_match.get('player1_name')}, player2={first_match.get('player2_name')}")

            # 2. Process each match
            for match_data in matches_raw:
                try:
                    result = self._process_match(match_data)

                    if result["created"]:
                        stats["matches_new"] += 1
                        stats["matches_created"].append(result["match_info"])
                        if result["prediction_generated"]:
                            stats["predictions_generated"] += 1
                    elif result.get("filtered"):
                        stats["matches_filtered"] += 1
                    else:
                        stats["matches_existing"] += 1

                except Exception as e:
                    error_msg = f"Error processing match {match_data.get('player1_name')} vs {match_data.get('player2_name')}: {e}"
                    logger.error(f"❌ {error_msg}")
                    stats["errors"].append(error_msg)

            # 3. Summary
            logger.info("=" * 60)
            logger.info("📊 Daily Match Fetch Summary:")
            logger.info(f"   Matches found: {stats['matches_found']}")
            logger.info(f"   New matches created: {stats['matches_new']}")
            logger.info(f"   Already existing: {stats['matches_existing']}")
            logger.info(f"   Filtered (WTA/doubles/etc): {stats['matches_filtered']}")
            logger.info(f"   Predictions generated: {stats['predictions_generated']}")
            logger.info(f"   API calls made: {stats['api_calls_made']}")
            logger.info(f"   Errors: {len(stats['errors'])}")
            logger.info("=" * 60)

            return stats

        except Exception as e:
            error_msg = f"Fatal error in daily match fetch: {e}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            stats["errors"].append(error_msg)
            return stats

    def fetch_matches_for_date(self, target_date: date) -> Dict:
        """
        Fetches matches for a specific date
        
        Útil para cargar datos históricos o fetchear días específicos.
        
        Args:
            target_date: Fecha específica para fetchear
            
        Returns:
            Dict con estadísticas de la operación
        """
        logger.info(f"📅 Fetching matches for {target_date}...")
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date.isoformat(),
            "matches_found": 0,
            "matches_new": 0,
            "matches_existing": 0,
            "predictions_generated": 0,
            "api_calls_made": 0,
            "errors": [],
        }
        
        try:
            # Fetch matches para esa fecha específica (1 día)
            date_str = target_date.strftime("%Y-%m-%d")
            
            # Usar el cliente API directamente con fecha específica
            logger.info(f"📥 Fetching fixtures for {date_str}...")
            
            # Calcular días desde hoy
            days_diff = (target_date - date.today()).days
            
            if days_diff < 0:
                # Fecha pasada - usar días negativos no funciona, usar fecha específica
                logger.info(f"ℹ️  Fetching historical date: {date_str}")
                # Modificar temporalmente para obtener esa fecha
                matches_raw = []
                params = {"date_start": date_str, "date_stop": date_str}
                data = self.api_client._make_request("get_fixtures", params)
                if data:
                    matches_raw = data.get("result", [])
                    
                # Obtener cuotas batch para esa fecha
                all_odds = self.api_client.get_all_odds_batch(date_str, date_str)
                
                # Procesar matches con cuotas
                matches_processed = []
                for match in matches_raw:
                    match_info = self.api_client.extract_match_info(match)
                    if match_info:
                        match_key = match_info["match_id"]
                        if match_key and str(match_key) in all_odds:
                            best_odds = self.api_client.extract_best_odds(all_odds, match_key)
                            if best_odds:
                                match_info.update(best_odds)
                        matches_processed.append(match_info)
                
                matches_raw = matches_processed
            else:
                # Fecha futura - usar el método normal
                matches_raw = self.api_client.get_all_matches_with_odds(days_ahead=max(days_diff, 1))
            
            stats["matches_found"] = len(matches_raw)
            stats["api_calls_made"] = self.api_client.requests_made
            
            # Procesar cada partido
            for match_data in matches_raw:
                try:
                    result = self._process_match(match_data)
                    
                    if result["created"]:
                        stats["matches_new"] += 1
                        if result["prediction_generated"]:
                            stats["predictions_generated"] += 1
                    else:
                        stats["matches_existing"] += 1
                        
                except Exception as e:
                    error_msg = f"Error processing match: {e}"
                    logger.error(f"❌ {error_msg}")
                    stats["errors"].append(error_msg)
            
            logger.info(f"✅ Fetch for {target_date} completed: {stats['matches_new']} new, {stats['matches_existing']} existing")
            return stats
            
        except Exception as e:
            error_msg = f"Error fetching matches for {target_date}: {e}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            stats["errors"].append(error_msg)
            return stats

    def sync_fixtures_for_dates(self, dates: Optional[List[date]] = None) -> Dict:
        """
        Sincroniza fixtures por fecha: obtiene get_fixtures para cada fecha y crea
        partidos que no existan. Sirve para no perder partidos que otras apps muestran.
        Por defecto sincroniza hoy y mañana.
        """
        if dates is None:
            dates = [date.today(), date.today() + timedelta(days=1)]
        stats = {
            "timestamp": datetime.now().isoformat(),
            "dates": [d.isoformat() for d in dates],
            "matches_found": 0,
            "matches_new": 0,
            "matches_existing": 0,
            "errors": [],
        }
        try:
            for target_date in dates:
                date_str = target_date.strftime("%Y-%m-%d")
                data = self.api_client._make_request("get_fixtures", {
                    "date_start": date_str,
                    "date_stop": date_str,
                })
                if not data or "result" not in data:
                    continue
                result = data["result"]
                raw_list = result if isinstance(result, list) else list(result.values()) if isinstance(result, dict) else []
                # Normalizar: cada elemento puede ser un partido (dict) o lista de uno
                matches_raw = []
                for m in raw_list:
                    if isinstance(m, list):
                        matches_raw.extend(m)
                    elif isinstance(m, dict):
                        matches_raw.append(m)
                all_odds = self.api_client.get_all_odds_batch(date_str, date_str) or {}
                for raw_match in matches_raw:
                    try:
                        match_info = self.api_client.extract_match_info(raw_match)
                        if not match_info:
                            continue
                        match_key = match_info.get("match_id") or match_info.get("event_key")
                        if match_key and str(match_key) in all_odds:
                            best = self.api_client.extract_best_odds(all_odds, match_key)
                            if best:
                                match_info.update(best)
                        stats["matches_found"] += 1
                        result = self._process_match(match_info)
                        if result.get("created"):
                            stats["matches_new"] += 1
                        elif not result.get("filtered"):
                            stats["matches_existing"] += 1
                    except Exception as e:
                        stats["errors"].append(str(e))
                        logger.debug(f"Error procesando partido {date_str}: {e}")
            if stats["matches_new"] > 0:
                logger.info(f"✅ Sync fixtures: {stats['matches_new']} nuevos, {stats['matches_existing']} ya existían")
            return stats
        except Exception as e:
            logger.error(f"❌ Error sync_fixtures_for_dates: {e}", exc_info=True)
            stats["errors"].append(str(e))
            return stats

    def _determine_match_status(
        self, event_live: str, event_final_result: str, event_status: str
    ) -> str:
        """
        Determina el estado del partido basándose en los datos de la API.
        
        Maneja todos los casos especiales: Finished, Walk Over, Retired, etc.
        """
        # PRIORIDAD 1: Si está en vivo
        if event_live == "1":
            return "en_juego"
        
        event_status_lower = (event_status or "").lower()
        
        # PRIORIDAD 2: Estados que indican que el partido terminó
        finished_keywords = [
            "finished", "ended", "completed", "final",
            "walk over", "walkover", "w.o.", "wo", "w/o",
            "retired", "ret", "retirement",
            "defaulted", "def", "default",
            "cancelled", "canceled", "postponed", "suspended",
            "awarded"
        ]
        
        for keyword in finished_keywords:
            if keyword in event_status_lower:
                return "completado"
        
        # PRIORIDAD 3: Si tiene resultado final válido
        if event_final_result and event_final_result != "-":
            return "completado"
        
        # PRIORIDAD 4: Si tiene algún status no vacío (podría estar en juego)
        if event_status and event_status_lower not in ["", "not started", "scheduled"]:
            return "en_juego"
        
        return "pendiente"

    def _process_match(self, match_data: Dict) -> Dict:
        """
        Processes a single match: checks if exists, creates if new, generates prediction

        Args:
            match_data: Match data from Tennis API

        Returns:
            Dict with processing result
        """
        # ===== FILTROS: ATP Singles + Challenger + ITF Men Singles (todos los masculinos individuales) =====
        
        # 0. VALIDACIÓN: Detectar tipo de evento
        event_type = (match_data.get("event_type") or match_data.get("event_type_type") or "").upper()
        league = (match_data.get("league") or "").upper()
        
        is_singles = "SINGLES" in event_type or "SINGLE" in event_type
        is_doubles = "DOUBLES" in event_type or "DOUBLE" in event_type
        
        # Rechazar dobles, júnior (Boys/Girls) y WTA
        if is_doubles or "BOYS" in event_type or "GIRLS" in event_type or "WTA" in event_type or "WOMEN" in event_type:
            logger.debug(f"⏭️  Ignorando (dobles/júnior/WTA): {event_type}")
            return {"created": False, "match_info": None, "prediction_generated": False, "filtered": True}
        
        # Aceptar: ATP Singles, Challenger Men Singles, ITF Men Singles (circuito masculino individual)
        is_atp_main = "ATP" in event_type or "ATP" in league
        is_challenger = "CHALLENGER" in event_type and "MEN" in event_type
        is_itf_men = "ITF" in event_type and "MEN" in event_type
        is_men_singles = "MEN" in event_type and is_singles
        
        if event_type:
            if not is_singles:
                logger.debug(f"⏭️  Ignorando (no singles): {event_type}")
                return {"created": False, "match_info": None, "prediction_generated": False, "filtered": True}
            if not (is_atp_main or is_challenger or is_itf_men or is_men_singles):
                logger.debug(f"⏭️  Ignorando tipo no masculino singles: {event_type}")
                return {"created": False, "match_info": None, "prediction_generated": False, "filtered": True}
        else:
            logger.debug(f"⚠️  Sin event_type, aplicando filtros de backup")
        
        # 1. BACKUP: Filtrar WTA (verificar múltiples campos)
        league = match_data.get("league", "").upper()
        tournament = match_data.get("tournament", "")
        tournament_lower = tournament.lower()
        
        # Detectar WTA / júnior / femenino (backup por si event_type falla)
        is_wta = (
            "WTA" in league or
            "WTA" in event_type or
            "WTA" in tournament or
            "WOMEN" in event_type or
            "WOMEN" in tournament_lower or
            "LADIES" in event_type or
            "LADIES" in tournament_lower or
            "FEMALE" in event_type or
            "FEMALE" in tournament_lower or
            "GIRLS" in event_type or
            "GIRLS" in tournament_lower or
            "BOYS" in event_type or
            "BOYS" in tournament_lower or
            # W-series tournaments - verificar inicio del nombre
            (tournament.startswith("W") and " " in tournament and tournament.split()[0][1:].isdigit())
        )
        
        if is_wta:
            logger.debug(f"⏭️  Ignorando partido WTA (backup filter): {tournament}")
            return {"created": False, "match_info": None, "prediction_generated": False, "filtered": True}
        
        # 2. BACKUP: Filtrar dobles (buscar "/" o "Doubles" en nombres)
        player1_name = match_data.get("player1_name", "Unknown")
        player2_name = match_data.get("player2_name", "Unknown")
        
        if "/" in player1_name or "/" in player2_name:
            logger.debug(f"⏭️  Ignorando partido de dobles: {player1_name} vs {player2_name}")
            return {"created": False, "match_info": None, "prediction_generated": False, "filtered": True}
        
        if "Doubles" in tournament or "doubles" in tournament:
            logger.debug(f"⏭️  Ignorando torneo de dobles: {tournament}")
            return {"created": False, "match_info": None, "prediction_generated": False, "filtered": True}
        
        # ===== Continuar con procesamiento normal =====
        
        match_date_str = match_data.get("date")

        # Parse date
        try:
            match_date = datetime.strptime(match_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(f"⚠️  Invalid date format: {match_date_str}, using today")
            match_date = date.today()

        # Check if match already exists
        if self.db.match_exists(player1_name, player2_name, match_date):
            logger.debug(f"ℹ️  Match already exists: {player1_name} vs {player2_name}")
            return {"created": False, "match_info": None, "prediction_generated": False, "filtered": False}

        # Determine surface
        surface = self.surface_mapper.get_surface(tournament)

        # Get odds (pueden ser None)
        player1_odds = match_data.get("player1_odds") or 0.0
        player2_odds = match_data.get("player2_odds") or 0.0
        
        has_odds = player1_odds > 0 and player2_odds > 0

        # Determinar estado inicial del partido basándose en datos de la API
        event_live = match_data.get("event_live", "0")
        event_final_result = match_data.get("event_final_result", "-")
        event_status = match_data.get("status", "")
        
        # Lógica de determinación de estado - PRIORIDAD: live > completado > pendiente
        estado_inicial = self._determine_match_status(event_live, event_final_result, event_status)

        # Create match in database (SIEMPRE, con o sin cuotas)
        try:
            match_id = self.db.create_match(
                fecha_partido=match_date,
                superficie=surface,
                jugador1_nombre=player1_name,
                jugador1_cuota=player1_odds,
                jugador2_nombre=player2_name,
                jugador2_cuota=player2_odds,
                hora_inicio=match_data.get("time"),
                torneo=tournament,
                ronda=match_data.get("round"),
                jugador1_ranking=None,
                jugador2_ranking=None,
                # Campos adicionales para tracking
                event_key=match_data.get("event_key"),
                jugador1_key=match_data.get("player1_key"),
                jugador2_key=match_data.get("player2_key"),
                tournament_key=match_data.get("tournament_key"),
                tournament_season=match_data.get("tournament_season"),
                event_live=event_live,
                event_qualification=match_data.get("event_qualification"),
                # Logos de jugadores
                jugador1_logo=match_data.get("player1_logo"),
                jugador2_logo=match_data.get("player2_logo"),
                # Estado inicial determinado
                estado=estado_inicial,
            )

            if has_odds:
                logger.info(
                    f"✅ Created match {match_id}: {player1_name} vs {player2_name} ({surface}) - Cuotas: {player1_odds}/{player2_odds}"
                )
            else:
                logger.info(
                    f"✅ Created match {match_id}: {player1_name} vs {player2_name} ({surface}) - SIN CUOTAS"
                )

            # Generate prediction SOLO si hay cuotas
            prediction_generated = False
            if has_odds:
                prediction_generated = self._generate_prediction(
                    match_id=match_id,
                    player1_name=player1_name,
                    player2_name=player2_name,
                    surface=surface,
                    player1_odds=player1_odds,
                    player2_odds=player2_odds,
                )
            else:
                logger.debug(f"ℹ️  Skipping prediction for match {match_id} (no odds)")

            return {
                "created": True,
                "match_info": {
                    "match_id": match_id,
                    "player1": player1_name,
                    "player2": player2_name,
                    "date": match_date.isoformat(),
                    "tournament": tournament,
                    "surface": surface,
                    "has_odds": has_odds,
                },
                "prediction_generated": prediction_generated,
            }

        except Exception as e:
            logger.error(f"❌ Error creating match: {e}")
            raise

    def _generate_prediction(
        self,
        match_id: int,
        player1_name: str,
        player2_name: str,
        surface: str,
        player1_odds: float,
        player2_odds: float,
    ) -> bool:
        """
        Generates prediction for a match (delega en lógica compartida).
        """
        from src.services.prediction_runner import run_prediction_and_save

        return run_prediction_and_save(
            db=self.db,
            predictor=self.predictor,
            match_id=match_id,
            player1_name=player1_name,
            player2_name=player2_name,
            surface=surface,
            player1_odds=player1_odds,
            player2_odds=player2_odds,
        )


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 60)
    print("🎾 Daily Match Fetcher - Standalone Execution")
    print("=" * 60 + "\n")

    # Initialize components
    db = MatchDatabase("matches_v2.db")
    api_client = APITennisClient()
    predictor = PredictorCalibrado(Config.MODEL_PATH)

    # Create fetcher
    fetcher = DailyMatchFetcher(db, api_client, predictor)

    # Fetch matches
    stats = fetcher.fetch_and_store_matches(days_ahead=7)

    # Print results
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    print(f"Matches found: {stats['matches_found']}")
    print(f"New matches created: {stats['matches_new']}")
    print(f"Already existing: {stats['matches_existing']}")
    print(f"Predictions generated: {stats['predictions_generated']}")
    print(f"API calls made: {stats['api_calls_made']}")
    print(f"Errors: {len(stats['errors'])}")

    if stats["errors"]:
        print("\n⚠️  Errors encountered:")
        for error in stats["errors"]:
            print(f"  - {error}")

    print("\n✅ Daily match fetch completed!")
    print("=" * 60 + "\n")

    db.close()
