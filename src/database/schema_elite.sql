-- Elite Tennis Analytics System - Database Schema
-- Version: 2.0 Elite
-- Created: 2026-01-15

-- ============================================================================
-- TABLA 1: PLAYERS - Perfiles de Jugadores
-- ============================================================================
CREATE TABLE IF NOT EXISTS players (
    player_key INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL,
    player_country TEXT,
    player_birthday TEXT,
    player_logo TEXT,
    atp_ranking INTEGER,
    wta_ranking INTEGER,
    ranking_movement TEXT CHECK(ranking_movement IN ('up', 'down', 'same')),
    ranking_points INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_players_name ON players(player_name);
CREATE INDEX IF NOT EXISTS idx_players_ranking ON players(atp_ranking, wta_ranking);

-- ============================================================================
-- TABLA 2: PLAYER_STATS - Estadísticas por Temporada
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_key INTEGER NOT NULL,
    season INTEGER NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('singles', 'doubles')),
    rank INTEGER,
    titles INTEGER DEFAULT 0,
    matches_won INTEGER DEFAULT 0,
    matches_lost INTEGER DEFAULT 0,
    hard_won INTEGER DEFAULT 0,
    hard_lost INTEGER DEFAULT 0,
    clay_won INTEGER DEFAULT 0,
    clay_lost INTEGER DEFAULT 0,
    grass_won INTEGER DEFAULT 0,
    grass_lost INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_key) REFERENCES players(player_key) ON DELETE CASCADE,
    UNIQUE(player_key, season, type)
);

CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_stats(player_key);
CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_stats(season);

-- ============================================================================
-- TABLA 3: TOURNAMENTS - Catálogo de Torneos
-- ============================================================================
CREATE TABLE IF NOT EXISTS tournaments (
    tournament_key INTEGER PRIMARY KEY,
    tournament_name TEXT NOT NULL,
    event_type_key INTEGER,
    event_type_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tournaments_name ON tournaments(tournament_name);
CREATE INDEX IF NOT EXISTS idx_tournaments_type ON tournaments(event_type_key);

-- ============================================================================
-- TABLA 4: MATCH_ODDS - Cuotas Multi-Bookmaker
-- ============================================================================
CREATE TABLE IF NOT EXISTS match_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    bookmaker TEXT NOT NULL,
    market_type TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_match_odds_match ON match_odds(match_id);
CREATE INDEX IF NOT EXISTS idx_match_odds_bookmaker ON match_odds(bookmaker);
CREATE INDEX IF NOT EXISTS idx_match_odds_market ON match_odds(market_type);
CREATE INDEX IF NOT EXISTS idx_match_odds_timestamp ON match_odds(timestamp);

-- ============================================================================
-- TABLA 5: MATCH_POINTBYPOINT - Punto por Punto
-- ============================================================================
CREATE TABLE IF NOT EXISTS match_pointbypoint (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    set_number TEXT NOT NULL,
    game_number INTEGER NOT NULL,
    point_number INTEGER NOT NULL,
    server TEXT CHECK(server IN ('First Player', 'Second Player')),
    score TEXT NOT NULL,
    is_break_point BOOLEAN DEFAULT 0,
    is_set_point BOOLEAN DEFAULT 0,
    is_match_point BOOLEAN DEFAULT 0,
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pbp_match ON match_pointbypoint(match_id);
CREATE INDEX IF NOT EXISTS idx_pbp_set ON match_pointbypoint(match_id, set_number);
CREATE INDEX IF NOT EXISTS idx_pbp_game ON match_pointbypoint(match_id, set_number, game_number);

-- ============================================================================
-- TABLA 6: MATCH_GAMES - Juegos del Partido
-- ============================================================================
CREATE TABLE IF NOT EXISTS match_games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    set_number TEXT NOT NULL,
    game_number INTEGER NOT NULL,
    server TEXT CHECK(server IN ('First Player', 'Second Player')),
    winner TEXT CHECK(winner IN ('First Player', 'Second Player')),
    score_games TEXT NOT NULL,
    score_sets TEXT NOT NULL,
    was_break BOOLEAN DEFAULT 0,
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    UNIQUE(match_id, set_number, game_number)
);

CREATE INDEX IF NOT EXISTS idx_games_match ON match_games(match_id);
CREATE INDEX IF NOT EXISTS idx_games_set ON match_games(match_id, set_number);
CREATE INDEX IF NOT EXISTS idx_games_break ON match_games(was_break);

-- ============================================================================
-- TABLA 7: HEAD_TO_HEAD - Histórico H2H
-- ============================================================================
CREATE TABLE IF NOT EXISTS head_to_head (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player1_key INTEGER NOT NULL,
    player2_key INTEGER NOT NULL,
    match_id INTEGER NOT NULL,
    match_date DATE NOT NULL,
    winner_key INTEGER NOT NULL,
    tournament_name TEXT,
    surface TEXT,
    final_result TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (player1_key) REFERENCES players(player_key),
    FOREIGN KEY (player2_key) REFERENCES players(player_key),
    FOREIGN KEY (winner_key) REFERENCES players(player_key)
);

CREATE INDEX IF NOT EXISTS idx_h2h_players ON head_to_head(player1_key, player2_key);
CREATE INDEX IF NOT EXISTS idx_h2h_match ON head_to_head(match_id);
CREATE INDEX IF NOT EXISTS idx_h2h_date ON head_to_head(match_date);

-- ============================================================================
-- MODIFICACIONES A TABLA MATCHES
-- ============================================================================

-- Agregar columnas nuevas a la tabla matches existente
ALTER TABLE matches ADD COLUMN first_player_key INTEGER;
ALTER TABLE matches ADD COLUMN second_player_key INTEGER;
ALTER TABLE matches ADD COLUMN tournament_key INTEGER;
ALTER TABLE matches ADD COLUMN tournament_season INTEGER;
ALTER TABLE matches ADD COLUMN event_qualification BOOLEAN DEFAULT 0;
ALTER TABLE matches ADD COLUMN event_game_result TEXT;
ALTER TABLE matches ADD COLUMN event_serve TEXT;
ALTER TABLE matches ADD COLUMN event_status_detail TEXT;

-- Crear índices para las nuevas columnas
CREATE INDEX IF NOT EXISTS idx_matches_player1 ON matches(first_player_key);
CREATE INDEX IF NOT EXISTS idx_matches_player2 ON matches(second_player_key);
CREATE INDEX IF NOT EXISTS idx_matches_tournament ON matches(tournament_key);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(tournament_season);

-- ============================================================================
-- VISTAS ÚTILES
-- ============================================================================

-- Vista de partidos con información completa de jugadores
CREATE VIEW IF NOT EXISTS v_matches_full AS
SELECT 
    m.*,
    p1.player_name as player1_full_name,
    p1.player_country as player1_country,
    p1.atp_ranking as player1_atp_ranking,
    p1.wta_ranking as player1_wta_ranking,
    p2.player_name as player2_full_name,
    p2.player_country as player2_country,
    p2.atp_ranking as player2_atp_ranking,
    p2.wta_ranking as player2_wta_ranking,
    t.tournament_name as tournament_full_name,
    t.event_type_type as tournament_type
FROM matches m
LEFT JOIN players p1 ON m.first_player_key = p1.player_key
LEFT JOIN players p2 ON m.second_player_key = p2.player_key
LEFT JOIN tournaments t ON m.tournament_key = t.tournament_key;

-- Vista de estadísticas de jugadores con porcentajes
CREATE VIEW IF NOT EXISTS v_player_stats_calculated AS
SELECT 
    ps.*,
    ROUND(CAST(ps.matches_won AS REAL) / NULLIF(ps.matches_won + ps.matches_lost, 0) * 100, 2) as win_percentage,
    ROUND(CAST(ps.hard_won AS REAL) / NULLIF(ps.hard_won + ps.hard_lost, 0) * 100, 2) as hard_win_percentage,
    ROUND(CAST(ps.clay_won AS REAL) / NULLIF(ps.clay_won + ps.clay_lost, 0) * 100, 2) as clay_win_percentage,
    ROUND(CAST(ps.grass_won AS REAL) / NULLIF(ps.grass_won + ps.grass_lost, 0) * 100, 2) as grass_win_percentage
FROM player_stats ps;

-- ============================================================================
-- TRIGGERS PARA MANTENER DATOS ACTUALIZADOS
-- ============================================================================

-- Trigger para actualizar last_updated en players
CREATE TRIGGER IF NOT EXISTS update_player_timestamp
AFTER UPDATE ON players
FOR EACH ROW
BEGIN
    UPDATE players SET last_updated = CURRENT_TIMESTAMP WHERE player_key = NEW.player_key;
END;

-- ============================================================================
-- DATOS INICIALES
-- ============================================================================

-- Insertar tipos de eventos comunes (se sincronizarán más desde la API)
INSERT OR IGNORE INTO tournaments (tournament_key, tournament_name, event_type_key, event_type_type)
VALUES 
    (0, 'Unknown', 0, 'Unknown');

-- ============================================================================
-- VERIFICACIÓN DE SCHEMA
-- ============================================================================

-- Verificar que todas las tablas se crearon correctamente
SELECT 
    name as table_name,
    sql as create_statement
FROM sqlite_master 
WHERE type = 'table' 
AND name IN ('players', 'player_stats', 'tournaments', 'match_odds', 
             'match_pointbypoint', 'match_games', 'head_to_head')
ORDER BY name;
