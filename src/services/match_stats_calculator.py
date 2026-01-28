"""
Match Stats Calculator - Calculador de Estadísticas de Partido
==============================================================

Servicio que calcula todas las estadísticas de un partido de tenis
a partir de los datos de la API Tennis (scores, pointbypoint).

Funcionalidades:
- Calcular scores estructurados
- Calcular estadísticas de saque/resto
- Calcular break points
- Generar timeline de juegos
- Identificar puntos clave
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.api.models_match_detail import (
    BreakPointStats,
    GameInfo,
    LiveScore,
    MatchScores,
    MatchStats,
    MatchTimeline,
    PlayerStats,
    PointByPointData,
    PointInfo,
    ReturnStats,
    ServeStats,
    SetScore,
    SetTimeline,
)

logger = logging.getLogger(__name__)


class MatchStatsCalculator:
    """
    Calculador de estadísticas de partido.
    
    Toma los datos crudos de la API Tennis y genera:
    - Scores estructurados
    - Estadísticas completas
    - Timeline de juegos
    - Puntos clave
    """
    
    def __init__(self):
        pass
    
    # ============================================================
    # SCORES
    # ============================================================
    
    def calculate_scores(
        self,
        api_scores: List[Dict],
        api_match: Optional[Dict] = None
    ) -> MatchScores:
        """
        Calcula los scores estructurados del partido.
        
        Args:
            api_scores: Lista de scores de la API Tennis
                [{"score_first": "6", "score_second": "4", "score_set": "1"}, ...]
            api_match: Datos completos del partido (para live data)
            
        Returns:
            MatchScores con sets detallados y datos en vivo
        """
        sets: List[SetScore] = []
        p1_sets = 0
        p2_sets = 0
        
        for score in api_scores:
            try:
                p1_games = int(score.get("score_first", 0))
                p2_games = int(score.get("score_second", 0))
                set_num = int(score.get("score_set", len(sets) + 1))
                
                # Determinar ganador del set
                winner = None
                if p1_games > p2_games:
                    winner = 1
                    p1_sets += 1
                elif p2_games > p1_games:
                    winner = 2
                    p2_sets += 1
                
                # Detectar tiebreak (si hay 7-6 o 6-7)
                tiebreak = None
                if (p1_games == 7 and p2_games == 6) or (p1_games == 6 and p2_games == 7):
                    # El tiebreak score vendría de pointbypoint si disponible
                    tiebreak = None  # Se llenará después si hay datos
                
                sets.append(SetScore(
                    set_number=set_num,
                    player1_games=p1_games,
                    player2_games=p2_games,
                    tiebreak_score=tiebreak,
                    winner=winner
                ))
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parseando score de set: {e}")
                continue
        
        # Datos en vivo si el partido está en juego
        live = None
        if api_match and api_match.get("event_live") == "1":
            game_result = api_match.get("event_game_result", "0-0")
            server = api_match.get("event_serve")
            status = api_match.get("event_status", "")
            
            live = LiveScore(
                current_game=game_result if game_result != "-" else "0-0",
                current_server=1 if server == "First Player" else 2,
                current_set=len(sets) + 1 if sets else 1,
                is_tiebreak="tiebreak" in status.lower() if status else False
            )
        
        return MatchScores(
            sets_won=[p1_sets, p2_sets],
            sets=sets,
            live=live
        )
    
    def parse_score_string(self, score_str: str) -> MatchScores:
        """
        Parsea un string de score como "6-4, 7-5, 6-3" a MatchScores.
        
        Args:
            score_str: String con el score (ej: "6-4, 7-5, 6-3" o "6-4 7-6(5) 6-3")
            
        Returns:
            MatchScores
        """
        sets = []
        p1_sets = 0
        p2_sets = 0
        
        if not score_str:
            return MatchScores(sets_won=[0, 0], sets=[])
        
        # Normalizar separadores
        parts = [p.strip() for p in score_str.replace(",", " ").split() if "-" in p or "(" in p]
        
        for i, part in enumerate(parts, 1):
            try:
                # Manejar tiebreak: "7-6(5)" o "7-6 (5)"
                tiebreak = None
                if "(" in part:
                    score_part = part.split("(")[0].strip()
                    tiebreak = part.split("(")[1].rstrip(")")
                else:
                    score_part = part
                
                if "-" in score_part:
                    p1_str, p2_str = score_part.split("-")
                    p1_games = int(p1_str.strip())
                    p2_games = int(p2_str.strip())
                    
                    winner = None
                    if p1_games > p2_games:
                        winner = 1
                        p1_sets += 1
                    elif p2_games > p1_games:
                        winner = 2
                        p2_sets += 1
                    
                    sets.append(SetScore(
                        set_number=i,
                        player1_games=p1_games,
                        player2_games=p2_games,
                        tiebreak_score=tiebreak,
                        winner=winner
                    ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parseando parte del score '{part}': {e}")
                continue
        
        return MatchScores(sets_won=[p1_sets, p2_sets], sets=sets)
    
    # ============================================================
    # TIMELINE
    # ============================================================
    
    def calculate_timeline(self, pointbypoint: List[Dict]) -> MatchTimeline:
        """
        Genera el timeline de juegos a partir del pointbypoint.
        
        Args:
            pointbypoint: Lista de juegos de la API Tennis
                [{"set_number": "Set 1", "number_game": "1", "player_served": "First Player", 
                  "serve_winner": "First Player", "score": "1 - 0", "points": [...]}, ...]
        
        Returns:
            MatchTimeline con todos los sets y juegos
        """
        if not pointbypoint:
            return MatchTimeline()
        
        # Agrupar juegos por set
        sets_data: Dict[int, List[Dict]] = {}
        
        for game in pointbypoint:
            try:
                # Parsear número de set ("Set 1" -> 1)
                set_str = game.get("set_number", "Set 1")
                if isinstance(set_str, str) and "Set" in set_str:
                    set_num = int(set_str.replace("Set", "").strip())
                else:
                    set_num = int(set_str) if set_str else 1
                
                if set_num not in sets_data:
                    sets_data[set_num] = []
                sets_data[set_num].append(game)
                
            except (ValueError, TypeError):
                continue
        
        # Construir timeline por set
        set_timelines = []
        total_games = 0
        total_breaks = 0
        
        for set_num in sorted(sets_data.keys()):
            games = sets_data[set_num]
            game_infos = []
            
            p1_games = 0
            p2_games = 0
            breaks_p1 = 0
            breaks_p2 = 0
            
            for game in games:
                try:
                    game_num = int(game.get("number_game", 0))
                    server_str = game.get("player_served", "First Player")
                    winner_str = game.get("serve_winner", server_str)
                    
                    server = 1 if "First" in server_str else 2
                    winner = 1 if "First" in winner_str else 2
                    
                    # Detectar break
                    is_break = bool(game.get("serve_lost")) or (server != winner)
                    
                    # Actualizar score
                    if winner == 1:
                        p1_games += 1
                        if is_break:
                            breaks_p1 += 1
                    else:
                        p2_games += 1
                        if is_break:
                            breaks_p2 += 1
                    
                    game_infos.append(GameInfo(
                        set_number=set_num,
                        game_number=game_num,
                        server=server,
                        winner=winner,
                        is_break=is_break,
                        score_after=f"{p1_games}-{p2_games}"
                    ))
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parseando juego: {e}")
                    continue
            
            # Determinar si hubo tiebreak
            has_tiebreak = (p1_games == 7 and p2_games == 6) or (p1_games == 6 and p2_games == 7)
            
            set_timelines.append(SetTimeline(
                set_number=set_num,
                games=game_infos,
                final_score=f"{p1_games}-{p2_games}",
                has_tiebreak=has_tiebreak,
                tiebreak_score=None,  # Se podría calcular del pointbypoint
                breaks_player1=breaks_p1,
                breaks_player2=breaks_p2
            ))
            
            total_games += len(game_infos)
            total_breaks += breaks_p1 + breaks_p2
        
        return MatchTimeline(
            sets=set_timelines,
            total_games=total_games,
            total_breaks=total_breaks,
            momentum_shifts=self._count_momentum_shifts(set_timelines)
        )
    
    def _count_momentum_shifts(self, sets: List[SetTimeline]) -> int:
        """Cuenta cambios de momentum (breaks consecutivos de diferentes jugadores)"""
        shifts = 0
        last_breaker = None
        
        for set_timeline in sets:
            for game in set_timeline.games:
                if game.is_break:
                    if last_breaker is not None and last_breaker != game.winner:
                        shifts += 1
                    last_breaker = game.winner
        
        return shifts
    
    # ============================================================
    # STATISTICS
    # ============================================================
    
    def calculate_stats(
        self,
        pointbypoint: List[Dict],
        scores: Optional[MatchScores] = None
    ) -> MatchStats:
        """
        Calcula todas las estadísticas del partido.
        
        Args:
            pointbypoint: Datos punto por punto de la API
            scores: Scores calculados (opcional, para total de sets)
            
        Returns:
            MatchStats con estadísticas completas
        """
        if not pointbypoint:
            return MatchStats(has_detailed_stats=False)
        
        # Inicializar contadores
        p1_stats = self._init_stats_counters()
        p2_stats = self._init_stats_counters()
        
        total_games = 0
        total_points = 0
        
        # Procesar cada juego
        for game in pointbypoint:
            server_str = game.get("player_served", "First Player")
            winner_str = game.get("serve_winner", server_str)
            is_server_p1 = "First" in server_str
            is_winner_p1 = "First" in winner_str
            
            # Juegos de saque
            if is_server_p1:
                p1_stats["service_games_total"] += 1
                if is_winner_p1:
                    p1_stats["service_games_won"] += 1
                else:
                    p2_stats["return_games_won"] += 1
                p2_stats["return_games_total"] += 1
            else:
                p2_stats["service_games_total"] += 1
                if not is_winner_p1:
                    p2_stats["service_games_won"] += 1
                else:
                    p1_stats["return_games_won"] += 1
                p1_stats["return_games_total"] += 1
            
            # Juegos ganados
            if is_winner_p1:
                p1_stats["games_won"] += 1
            else:
                p2_stats["games_won"] += 1
            
            total_games += 1
            
            # Procesar puntos del juego
            points = game.get("points", [])
            for point in points:
                total_points += 1
                
                # Break points
                if point.get("break_point"):
                    if is_server_p1:
                        p2_stats["bp_total"] += 1
                        p1_stats["bp_faced"] += 1
                    else:
                        p1_stats["bp_total"] += 1
                        p2_stats["bp_faced"] += 1
        
        # Calcular breaks desde juegos (servidor != ganador)
        for game in pointbypoint:
            server_str = game.get("player_served", "First Player")
            winner_str = game.get("serve_winner", server_str)
            is_server_p1 = "First" in server_str
            is_winner_p1 = "First" in winner_str
            
            if game.get("serve_lost") or (is_server_p1 != is_winner_p1):
                if is_winner_p1:
                    p1_stats["bp_won"] += 1
                else:
                    p2_stats["bp_won"] += 1
        
        # Calcular BP salvados
        p1_stats["bp_saved"] = p1_stats["bp_faced"] - p2_stats["return_games_won"]
        p2_stats["bp_saved"] = p2_stats["bp_faced"] - p1_stats["return_games_won"]
        
        # Construir PlayerStats
        player1_stats = self._build_player_stats(p1_stats, total_points)
        player2_stats = self._build_player_stats(p2_stats, total_points)
        
        return MatchStats(
            player1=player1_stats,
            player2=player2_stats,
            total_games=total_games,
            total_points=total_points,
            has_detailed_stats=True
        )
    
    def _init_stats_counters(self) -> Dict[str, int]:
        """Inicializa contadores de estadísticas"""
        return {
            "aces": 0,
            "double_faults": 0,
            "service_games_won": 0,
            "service_games_total": 0,
            "return_games_won": 0,
            "return_games_total": 0,
            "games_won": 0,
            "points_won": 0,
            "bp_won": 0,
            "bp_total": 0,
            "bp_saved": 0,
            "bp_faced": 0,
        }
    
    def _build_player_stats(self, counters: Dict[str, int], total_points: int) -> PlayerStats:
        """Construye PlayerStats desde contadores"""
        return PlayerStats(
            serve=ServeStats(
                aces=counters["aces"],
                double_faults=counters["double_faults"],
                service_games_won=counters["service_games_won"],
                service_games_total=counters["service_games_total"],
            ),
            return_=ReturnStats(
                return_games_won=counters["return_games_won"],
                return_games_total=counters["return_games_total"],
            ),
            break_points=BreakPointStats(
                break_points_won=counters["bp_won"],
                break_points_total=counters["bp_total"],
                break_points_saved=counters["bp_saved"],
                break_points_faced=counters["bp_faced"],
            ),
            total_games_won=counters["games_won"],
            total_points_won=counters["points_won"],
        )
    
    # ============================================================
    # POINT BY POINT
    # ============================================================
    
    def extract_point_by_point(
        self,
        pointbypoint: List[Dict],
        set_filter: Optional[int] = None
    ) -> PointByPointData:
        """
        Extrae datos punto por punto estructurados.
        
        Args:
            pointbypoint: Datos de la API
            set_filter: Filtrar por número de set (opcional)
            
        Returns:
            PointByPointData
        """
        points: List[PointInfo] = []
        key_points: List[PointInfo] = []
        
        for game in pointbypoint:
            try:
                set_str = game.get("set_number", "Set 1")
                if isinstance(set_str, str) and "Set" in set_str:
                    set_num = int(set_str.replace("Set", "").strip())
                else:
                    set_num = int(set_str) if set_str else 1
                
                # Filtrar por set si se especifica
                if set_filter is not None and set_num != set_filter:
                    continue
                
                game_num = int(game.get("number_game", 0))
                server_str = game.get("player_served", "First Player")
                server = 1 if "First" in server_str else 2
                
                game_points = game.get("points", [])
                for point in game_points:
                    point_num = int(point.get("number_point", 0))
                    score = point.get("score", "0-0")
                    is_bp = bool(point.get("break_point"))
                    is_sp = bool(point.get("set_point"))
                    is_mp = bool(point.get("match_point"))
                    
                    point_info = PointInfo(
                        set_number=set_num,
                        game_number=game_num,
                        point_number=point_num,
                        score=score,
                        server=server,
                        is_break_point=is_bp,
                        is_set_point=is_sp,
                        is_match_point=is_mp,
                    )
                    
                    points.append(point_info)
                    
                    if is_bp or is_sp or is_mp:
                        key_points.append(point_info)
                        
            except (ValueError, TypeError) as e:
                logger.warning(f"Error extrayendo punto: {e}")
                continue
        
        return PointByPointData(
            total_points=len(points),
            points=points,
            key_points=key_points
        )
    
    # ============================================================
    # UTILITY
    # ============================================================
    
    def estimate_duration(self, total_games: int) -> int:
        """
        Estima la duración del partido en minutos.
        
        Aproximación: ~4-5 minutos por juego en promedio.
        """
        return total_games * 4 + 10  # Base de 10 minutos + 4 por juego
