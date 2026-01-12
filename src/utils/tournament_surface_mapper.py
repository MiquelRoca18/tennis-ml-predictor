"""
Tournament Surface Mapper
==========================

Maps tournament names to their surface types since the Tennis API
doesn't provide surface information directly.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TournamentSurfaceMapper:
    """
    Maps tournament names to surface types (Hard, Clay, Grass, Carpet)
    """

    # Known tournament-to-surface mappings
    TOURNAMENT_SURFACES = {
        # Grand Slams
        "Australian Open": "Hard",
        "Roland Garros": "Clay",
        "French Open": "Clay",
        "Wimbledon": "Grass",
        "US Open": "Hard",
        # ATP Masters 1000
        "Indian Wells": "Hard",
        "Miami": "Hard",
        "Monte Carlo": "Clay",
        "Madrid": "Clay",
        "Rome": "Clay",
        "Canadian Open": "Hard",
        "Cincinnati": "Hard",
        "Shanghai": "Hard",
        "Paris": "Hard",
        # ATP 500
        "Rotterdam": "Hard",
        "Dubai": "Hard",
        "Acapulco": "Hard",
        "Barcelona": "Clay",
        "Halle": "Grass",
        "Queen's": "Grass",
        "Hamburg": "Clay",
        "Washington": "Hard",
        "Beijing": "Hard",
        "Tokyo": "Hard",
        "Basel": "Hard",
        "Vienna": "Hard",
        # Other notable tournaments
        "ATP Finals": "Hard",
        "Next Gen ATP Finals": "Hard",
        "Davis Cup": "Hard",  # Variable, but most common
        "Laver Cup": "Hard",
        "United Cup": "Hard",
        # Common keywords for surface detection
        "clay": "Clay",
        "grass": "Grass",
        "hard": "Hard",
        "indoor": "Hard",  # Most indoor tournaments are hard court
    }

    @classmethod
    def get_surface(cls, tournament_name: Optional[str]) -> str:
        """
        Determines the surface for a given tournament

        Args:
            tournament_name: Name of the tournament

        Returns:
            Surface type: "Hard", "Clay", "Grass", or "Carpet"
            Defaults to "Hard" if unknown (most common surface)
        """
        if not tournament_name:
            logger.debug("No tournament name provided, defaulting to Hard")
            return "Hard"

        tournament_lower = tournament_name.lower()

        # Check exact matches first
        for known_tournament, surface in cls.TOURNAMENT_SURFACES.items():
            if known_tournament.lower() in tournament_lower:
                logger.debug(f"Tournament '{tournament_name}' mapped to {surface}")
                return surface

        # Check for keywords in tournament name
        if "clay" in tournament_lower or "terre" in tournament_lower:
            logger.debug(f"Tournament '{tournament_name}' contains 'clay' keyword -> Clay")
            return "Clay"

        if "grass" in tournament_lower or "lawn" in tournament_lower:
            logger.debug(f"Tournament '{tournament_name}' contains 'grass' keyword -> Grass")
            return "Grass"

        # Default to Hard (most common surface ~70% of ATP tour)
        logger.debug(
            f"Tournament '{tournament_name}' unknown, defaulting to Hard (most common)"
        )
        return "Hard"

    @classmethod
    def get_surface_by_date(cls, tournament_name: Optional[str], date_str: str) -> str:
        """
        Determines surface considering both tournament name and date

        Some tournaments change surface or there are seasonal patterns:
        - January-March: Australian summer hard court season
        - April-June: European clay season
        - June-July: Grass season (Wimbledon)
        - August-November: North American/Asian hard court season

        Args:
            tournament_name: Name of the tournament
            date_str: Date in format YYYY-MM-DD

        Returns:
            Surface type
        """
        # First try to get from tournament name
        surface = cls.get_surface(tournament_name)

        # If unknown and we have a date, use seasonal heuristics
        if surface == "Hard" and not tournament_name:
            try:
                month = int(date_str.split("-")[1])

                # European clay season (April-June)
                if 4 <= month <= 6:
                    logger.debug(f"Date {date_str} in clay season, suggesting Clay")
                    # Don't override, just log - still return Hard as default

            except (ValueError, IndexError):
                pass

        return surface


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    mapper = TournamentSurfaceMapper()

    # Test cases
    test_tournaments = [
        "Australian Open",
        "Roland Garros",
        "Wimbledon",
        "US Open",
        "ATP Masters 1000 Monte Carlo",
        "Barcelona Open",
        "Unknown Tournament",
        None,
        "Indoor Championship",
        "Clay Court Championship",
    ]

    print("\n🎾 Tournament Surface Mapper Test\n")
    print("=" * 50)

    for tournament in test_tournaments:
        surface = mapper.get_surface(tournament)
        print(f"Tournament: {tournament or 'None':30} -> Surface: {surface}")

    print("\n" + "=" * 50)
    print("✅ Test completed!")
