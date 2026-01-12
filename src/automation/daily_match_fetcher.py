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
        predictor: PredictorCalibrado,
    ):
        """
        Initialize the daily match fetcher

        Args:
            db: Database instance
            api_client: Tennis API client
            predictor: ML predictor instance
        """
        self.db = db
        self.api_client = api_client
        self.predictor = predictor
        self.surface_mapper = TournamentSurfaceMapper()

        logger.info("✅ DailyMatchFetcher initialized")

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

            # 2. Process each match
            for match_data in matches_raw:
                try:
                    result = self._process_match(match_data)

                    if result["created"]:
                        stats["matches_new"] += 1
                        stats["matches_created"].append(result["match_info"])
                        if result["prediction_generated"]:
                            stats["predictions_generated"] += 1
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

    def _process_match(self, match_data: Dict) -> Dict:
        """
        Processes a single match: checks if exists, creates if new, generates prediction

        Args:
            match_data: Match data from Tennis API

        Returns:
            Dict with processing result
        """
        player1_name = match_data.get("player1_name", "Unknown")
        player2_name = match_data.get("player2_name", "Unknown")
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
            return {"created": False, "match_info": None, "prediction_generated": False}

        # Determine surface
        tournament = match_data.get("tournament", "Unknown")
        surface = self.surface_mapper.get_surface(tournament)

        # Get odds
        player1_odds = match_data.get("player1_odds")
        player2_odds = match_data.get("player2_odds")

        if not player1_odds or not player2_odds:
            logger.warning(
                f"⚠️  Missing odds for {player1_name} vs {player2_name}, skipping"
            )
            return {"created": False, "match_info": None, "prediction_generated": False}

        # Create match in database
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
                jugador1_ranking=None,  # Tennis API doesn't provide rankings
                jugador2_ranking=None,
            )

            logger.info(
                f"✅ Created match {match_id}: {player1_name} vs {player2_name} ({surface})"
            )

            # Generate prediction
            prediction_generated = self._generate_prediction(
                match_id=match_id,
                player1_name=player1_name,
                player2_name=player2_name,
                surface=surface,
                player1_odds=player1_odds,
                player2_odds=player2_odds,
            )

            return {
                "created": True,
                "match_info": {
                    "match_id": match_id,
                    "player1": player1_name,
                    "player2": player2_name,
                    "date": match_date.isoformat(),
                    "tournament": tournament,
                    "surface": surface,
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
        Generates prediction for a match

        Args:
            match_id: Database match ID
            player1_name: Name of player 1
            player2_name: Name of player 2
            surface: Surface type
            player1_odds: Odds for player 1
            player2_odds: Odds for player 2

        Returns:
            True if prediction was generated successfully
        """
        try:
            # Generate prediction using ML model
            resultado_pred = self.predictor.predecir_partido(
                jugador1=player1_name,
                jugador1_rank=999,  # Unknown ranking
                jugador2=player2_name,
                jugador2_rank=999,
                superficie=surface,
                cuota=player1_odds,
            )

            # Calculate probabilities and EVs
            prob_j1 = resultado_pred["probabilidad"]
            prob_j2 = 1 - prob_j1
            ev_j1 = resultado_pred["expected_value"]
            ev_j2 = (prob_j2 * player2_odds) - 1
            edge_j1 = resultado_pred.get("edge", 0)
            edge_j2 = prob_j2 - (1 / player2_odds)

            # Determine recommendation
            umbral_ev = Config.EV_THRESHOLD
            if ev_j1 > umbral_ev:
                recomendacion = f"APOSTAR a {player1_name}"
                mejor_opcion = player1_name
                kelly_j1 = resultado_pred.get("stake_recomendado", 0)
                kelly_j2 = None
            elif ev_j2 > umbral_ev:
                recomendacion = f"APOSTAR a {player2_name}"
                mejor_opcion = player2_name
                kelly_j1 = None
                kelly_pct = (prob_j2 * player2_odds - 1) / (player2_odds - 1)
                kelly_j2 = round(kelly_pct * 0.25 * 100, 2)
            else:
                recomendacion = "NO APOSTAR"
                mejor_opcion = None
                kelly_j1 = None
                kelly_j2 = None

            # Determine confidence
            if abs(prob_j1 - 0.5) > 0.15:
                confianza = "Alta"
            elif abs(prob_j1 - 0.5) > 0.08:
                confianza = "Media"
            else:
                confianza = "Baja"

            # Save prediction to database
            self.db.add_prediction(
                match_id=match_id,
                jugador1_cuota=player1_odds,
                jugador2_cuota=player2_odds,
                jugador1_probabilidad=prob_j1,
                jugador2_probabilidad=prob_j2,
                jugador1_ev=ev_j1,
                jugador2_ev=ev_j2,
                jugador1_edge=edge_j1,
                jugador2_edge=edge_j2,
                recomendacion=recomendacion,
                mejor_opcion=mejor_opcion,
                confianza=confianza,
                kelly_stake_jugador1=kelly_j1,
                kelly_stake_jugador2=kelly_j2,
            )

            logger.info(f"✅ Prediction generated for match {match_id}: {recomendacion}")
            return True

        except Exception as e:
            logger.error(f"❌ Error generating prediction for match {match_id}: {e}")
            return False


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
