"""
Player Name Normalization Utility
==================================

Normalizes player names from API-Tennis to match historical data format.

API-Tennis may return:
- "Alcaraz" or "C. Alcaraz" 
- "Sinner" or "J. Sinner"

Historical data has:
- "Carlos Alcaraz"
- "Jannik Sinner"

This module provides fuzzy matching to find the best match in historical data.
"""

import logging
from typing import Optional, Set
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class PlayerNameNormalizer:
    """
    Normalizes player names from API to match historical data format
    """
    
    def __init__(self, historical_player_names: Set[str]):
        """
        Initialize with set of known player names from historical data
        
        Args:
            historical_player_names: Set of full player names from historical data
        """
        self.known_players = historical_player_names
        self.cache = {}  # Cache for performance
        
        logger.info(f"✅ PlayerNameNormalizer initialized with {len(self.known_players)} known players")
    
    def normalize(self, api_name: str) -> str:
        """
        Normalize an API player name to match historical data format
        
        Args:
            api_name: Player name from API (e.g., "Alcaraz", "C. Alcaraz")
        
        Returns:
            Full player name from historical data (e.g., "Carlos Alcaraz")
            or original name if no match found
        """
        # Check cache first
        if api_name in self.cache:
            return self.cache[api_name]
        
        # If exact match exists, use it
        if api_name in self.known_players:
            self.cache[api_name] = api_name
            return api_name
        
        # Try fuzzy matching
        # Strategy 1: Find players whose last name matches
        last_name = api_name.split()[-1]  # Get last word as last name
        
        candidates = [
            name for name in self.known_players
            if last_name.lower() in name.lower()
        ]
        
        if len(candidates) == 1:
            # Unique match found
            matched_name = candidates[0]
            logger.debug(f"🔍 Matched '{api_name}' → '{matched_name}' (last name match)")
            self.cache[api_name] = matched_name
            return matched_name
        
        elif len(candidates) > 1:
            # Multiple candidates, use fuzzy matching to find best
            matches = get_close_matches(api_name, candidates, n=1, cutoff=0.6)
            if matches:
                matched_name = matches[0]
                logger.debug(f"🔍 Matched '{api_name}' → '{matched_name}' (fuzzy match)")
                self.cache[api_name] = matched_name
                return matched_name
        
        # No match found, return original and warn
        logger.warning(f"⚠️  No match found for player '{api_name}' in historical data")
        self.cache[api_name] = api_name
        return api_name
    
    def normalize_batch(self, api_names: list) -> dict:
        """
        Normalize multiple player names at once
        
        Args:
            api_names: List of player names from API
        
        Returns:
            Dict mapping API names to normalized names
        """
        return {name: self.normalize(name) for name in api_names}
