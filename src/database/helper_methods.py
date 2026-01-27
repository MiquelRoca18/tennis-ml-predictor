    # ============================================================
    # HELPER METHODS FOR SERVICES (PostgreSQL compatible)
    # ============================================================

    def update_match_hora_inicio(self, match_id: int, hora_inicio: str) -> bool:
        """
        Actualiza la hora de inicio de un partido
        
        Args:
            match_id: ID del partido
            hora_inicio: Nueva hora de inicio
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            self._execute(
                "UPDATE matches SET hora_inicio = :hora WHERE id = :match_id",
                {"hora": hora_inicio, "match_id": match_id}
            )
            return True
        except Exception as e:
            logger.error(f"Error actualizando hora de inicio: {e}")
            return False

    def update_match_ganador(self, match_id: int, ganador: str) -> bool:
        """
        Actualiza el ganador de un partido
        
        Args:
            match_id: ID del partido
            ganador: Nombre del ganador
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            self._execute(
                "UPDATE matches SET resultado_ganador = :ganador WHERE id = :match_id",
                {"ganador": ganador, "match_id": match_id}
            )
            return True
        except Exception as e:
            logger.error(f"Error actualizando ganador: {e}")
            return False

    def update_match_player_keys(self, match_id: int, player1_key: str, player2_key: str) -> bool:
        """
        Actualiza los player_keys de un partido
        
        Args:
            match_id: ID del partido
            player1_key: Key del jugador 1
            player2_key: Key del jugador 2
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            self._execute(
                """
                UPDATE matches
                SET jugador1_key = :player1_key, jugador2_key = :player2_key
                WHERE id = :match_id
            """,
                {"player1_key": player1_key, "player2_key": player2_key, "match_id": match_id}
            )
            return True
        except Exception as e:
            logger.error(f"Error actualizando player keys: {e}")
            return False

    def check_match_games_exist(self, match_id: int) -> int:
        """
        Verifica si ya existen juegos guardados para un partido
        
        Args:
            match_id: ID del partido
            
        Returns:
            Número de juegos existentes
        """
        result = self._fetchone(
            "SELECT COUNT(*) as count FROM match_games WHERE match_id = :match_id",
            {"match_id": match_id}
        )
        return result["count"] if result else 0

    def save_match_game(self, match_id: int, game_data: Dict) -> bool:
        """
        Guarda un juego de un partido
        
        Args:
            match_id: ID del partido
            game_data: Datos del juego
            
        Returns:
            True si se guardó correctamente
        """
        try:
            self._execute(
                """
                INSERT INTO match_games (
                    match_id, set_number, game_number,
                    server, winner, score_games, score_sets, was_break
                ) VALUES (
                    :match_id, :set_number, :game_number,
                    :server, :winner, :score_games, :score_sets, :was_break
                )
            """,
                {
                    "match_id": match_id,
                    "set_number": game_data.get("set_number", ""),
                    "game_number": game_data.get("number_game", ""),
                    "server": game_data.get("player_served", ""),
                    "winner": game_data.get("serve_winner", ""),
                    "score_games": game_data.get("score", ""),
                    "score_sets": "",
                    "was_break": 1 if game_data.get("serve_lost") else 0
                }
            )
            return True
        except Exception as e:
            logger.debug(f"Error guardando juego: {e}")
            return False

    def save_match_point(self, match_id: int, set_number: str, game_number: int, point_data: Dict) -> bool:
        """
        Guarda un punto de un partido
        
        Args:
            match_id: ID del partido
            set_number: Número de set
            game_number: Número de juego
            point_data: Datos del punto
            
        Returns:
            True si se guardó correctamente
        """
        try:
            self._execute(
                """
                INSERT INTO match_pointbypoint (
                    match_id, set_number, game_number, point_number,
                    server, score, is_break_point, is_set_point, is_match_point
                ) VALUES (
                    :match_id, :set_number, :game_number, :point_number,
                    :server, :score, :is_break_point, :is_set_point, :is_match_point
                )
            """,
                {
                    "match_id": match_id,
                    "set_number": set_number,
                    "game_number": game_number,
                    "point_number": point_data.get("number_point", ""),
                    "server": point_data.get("player_served", ""),
                    "score": point_data.get("score", ""),
                    "is_break_point": 1 if point_data.get("break_point") else 0,
                    "is_set_point": 1 if point_data.get("set_point") else 0,
                    "is_match_point": 1 if point_data.get("match_point") else 0
                }
            )
            return True
        except Exception as e:
            logger.debug(f"Error guardando punto: {e}")
            return False
