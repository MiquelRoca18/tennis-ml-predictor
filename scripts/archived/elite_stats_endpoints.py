# ============================================================================
# ENDPOINTS ELITE - DAY 2: ESTADÍSTICAS DETALLADAS
# ============================================================================

@app.get("/matches/{match_id}/stats/detailed", tags=["Elite - Match Statistics"])
async def get_match_detailed_stats(match_id: int):
    """
    Obtiene estadísticas detalladas completas de un partido
    Incluye: juegos, puntos, breaks, y análisis
    """
    if not pointbypoint_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        # Obtener juegos
        games = pointbypoint_service.get_match_games(match_id)
        
        # Obtener puntos
        points = pointbypoint_service.get_match_points(match_id)
        
        # Calcular estadísticas de breaks
        break_stats = pointbypoint_service.get_break_point_stats(match_id)
        
        # Calcular estadísticas adicionales
        total_games = len(games) if games else 0
        total_points = len(points) if points else 0
        
        # Contar breaks
        breaks_player1 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'First Player') if games else 0
        breaks_player2 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'Second Player') if games else 0
        
        # Organizar por sets
        sets_data = {}
        if games:
            for game in games:
                set_num = game.get('set_number', 'Unknown')
                if set_num not in sets_data:
                    sets_data[set_num] = {
                        'games': [],
                        'games_player1': 0,
                        'games_player2': 0
                    }
                sets_data[set_num]['games'].append(game)
                if game.get('winner') == 'First Player':
                    sets_data[set_num]['games_player1'] += 1
                elif game.get('winner') == 'Second Player':
                    sets_data[set_num]['games_player2'] += 1
        
        return {
            "match_id": match_id,
            "summary": {
                "total_games": total_games,
                "total_points": total_points,
                "breaks_player1": breaks_player1,
                "breaks_player2": breaks_player2,
                "total_sets": len(sets_data)
            },
            "sets": sets_data,
            "break_points": break_stats,
            "games": games,
            "points": points
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas detalladas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/stats/games", tags=["Elite - Match Statistics"])
async def get_match_games(match_id: int):
    """
    Obtiene el desglose de juegos de un partido
    """
    if not pointbypoint_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        games = pointbypoint_service.get_match_games(match_id)
        
        if not games:
            return {
                "match_id": match_id,
                "games": [],
                "message": "No hay datos de juegos disponibles para este partido"
            }
        
        return {
            "match_id": match_id,
            "total_games": len(games),
            "games": games
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo juegos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/stats/points", tags=["Elite - Match Statistics"])
async def get_match_points(match_id: int):
    """
    Obtiene todos los puntos jugados en un partido
    """
    if not pointbypoint_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        points = pointbypoint_service.get_match_points(match_id)
        
        if not points:
            return {
                "match_id": match_id,
                "points": [],
                "message": "No hay datos punto por punto disponibles para este partido"
            }
        
        # Organizar por sets y juegos
        organized = {}
        for point in points:
            set_num = point.get('set_number', 'Unknown')
            game_num = point.get('game_number', 'Unknown')
            
            if set_num not in organized:
                organized[set_num] = {}
            if game_num not in organized[set_num]:
                organized[set_num][game_num] = []
            
            organized[set_num][game_num].append(point)
        
        return {
            "match_id": match_id,
            "total_points": len(points),
            "points_by_set": organized,
            "all_points": points
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo puntos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/stats/breaks", tags=["Elite - Match Statistics"])
async def get_match_break_points(match_id: int):
    """
    Obtiene estadísticas de break points de un partido
    """
    if not pointbypoint_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        break_stats = pointbypoint_service.get_break_point_stats(match_id)
        
        return {
            "match_id": match_id,
            "break_points": break_stats
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de breaks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matches/{match_id}/stats/summary", tags=["Elite - Match Statistics"])
async def get_match_stats_summary(match_id: int):
    """
    Obtiene un resumen de estadísticas calculadas del partido
    """
    if not pointbypoint_service:
        raise HTTPException(status_code=503, detail="PointByPoint service not available")
    
    try:
        games = pointbypoint_service.get_match_games(match_id)
        points = pointbypoint_service.get_match_points(match_id)
        
        if not games or not points:
            return {
                "match_id": match_id,
                "message": "Estadísticas no disponibles para este partido"
            }
        
        # Calcular estadísticas
        total_games = len(games)
        total_points = len(points)
        
        # Juegos ganados
        games_won_p1 = sum(1 for g in games if g.get('winner') == 'First Player')
        games_won_p2 = sum(1 for g in games if g.get('winner') == 'Second Player')
        
        # Breaks
        breaks_p1 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'First Player')
        breaks_p2 = sum(1 for g in games if g.get('was_break') and g.get('winner') == 'Second Player')
        
        # Break points
        bp_total = sum(1 for p in points if p.get('is_break_point'))
        bp_p1 = sum(1 for p in points if p.get('is_break_point') and p.get('server') == 'Second Player')
        bp_p2 = sum(1 for p in points if p.get('is_break_point') and p.get('server') == 'First Player')
        
        # Puntos ganados al saque
        serve_points_p1 = sum(1 for p in points if p.get('server') == 'First Player')
        serve_points_p2 = sum(1 for p in points if p.get('server') == 'Second Player')
        
        return {
            "match_id": match_id,
            "summary": {
                "total_games": total_games,
                "total_points": total_points,
                "player1": {
                    "games_won": games_won_p1,
                    "breaks": breaks_p1,
                    "break_points": bp_p1,
                    "serve_points": serve_points_p1,
                    "break_conversion": f"{(breaks_p1/bp_p1*100):.1f}%" if bp_p1 > 0 else "N/A"
                },
                "player2": {
                    "games_won": games_won_p2,
                    "breaks": breaks_p2,
                    "break_points": bp_p2,
                    "serve_points": serve_points_p2,
                    "break_conversion": f"{(breaks_p2/bp_p2*100):.1f}%" if bp_p2 > 0 else "N/A"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculando resumen de estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
