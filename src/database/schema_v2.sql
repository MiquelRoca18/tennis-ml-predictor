-- ============================================================
-- Tennis ML Predictor v2.0 - Database Schema
-- ============================================================
-- Sistema de gestión de partidos con predicciones versionadas,
-- tracking de apuestas y historial de cuotas
-- ============================================================

-- Tabla principal de partidos
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- IDs de API-Tennis (para tracking y sincronización)
    event_key VARCHAR(50),  -- ID único del partido en API-Tennis
    jugador1_key VARCHAR(50),  -- ID del jugador 1 en API-Tennis
    jugador2_key VARCHAR(50),  -- ID del jugador 2 en API-Tennis
    tournament_key VARCHAR(50),  -- ID del torneo en API-Tennis
    
    -- Información del partido
    fecha_partido DATE NOT NULL,
    hora_inicio TIME,
    torneo VARCHAR(200),
    tournament_season VARCHAR(10),  -- Ej: "2026"
    ronda VARCHAR(100),
    superficie VARCHAR(20) NOT NULL CHECK(superficie IN ('Hard', 'Clay', 'Grass', 'Carpet')),
    
    -- Jugadores
    jugador1_nombre VARCHAR(200) NOT NULL,
    jugador1_ranking INTEGER,
    jugador1_logo TEXT,  -- URL del logo del jugador 1 desde API-Tennis
    jugador2_nombre VARCHAR(200) NOT NULL,
    jugador2_ranking INTEGER,
    jugador2_logo TEXT,  -- URL del logo del jugador 2 desde API-Tennis
    
    -- Estado en vivo
    event_live VARCHAR(1) DEFAULT '0',  -- '0' = no live, '1' = en vivo
    event_qualification VARCHAR(10) DEFAULT 'False',  -- Si es clasificación
    
    -- Resultado (NULL si no ha terminado)
    resultado_ganador VARCHAR(200),  -- Nombre del ganador
    resultado_marcador VARCHAR(100),  -- Ej: "6-4, 7-5, 6-3"
    event_final_result VARCHAR(20),  -- Ej: "2-0", "2-1"
    
    -- Estado del partido
    estado VARCHAR(20) NOT NULL DEFAULT 'pendiente' CHECK(estado IN ('pendiente', 'en_juego', 'completado', 'cancelado')),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Índices para búsquedas rápidas
    UNIQUE(fecha_partido, jugador1_nombre, jugador2_nombre)
);

CREATE INDEX idx_matches_fecha ON matches(fecha_partido);
CREATE INDEX idx_matches_estado ON matches(estado);
CREATE INDEX idx_matches_fecha_estado ON matches(fecha_partido, estado);
CREATE INDEX idx_matches_event_key ON matches(event_key);


-- Tabla de predicciones versionadas
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    
    -- Versionado
    version INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Cuotas en el momento de la predicción
    jugador1_cuota REAL NOT NULL,
    jugador2_cuota REAL NOT NULL,
    
    -- Probabilidades del modelo
    jugador1_probabilidad REAL NOT NULL CHECK(jugador1_probabilidad >= 0 AND jugador1_probabilidad <= 1),
    jugador2_probabilidad REAL NOT NULL CHECK(jugador2_probabilidad >= 0 AND jugador2_probabilidad <= 1),
    
    -- Expected Value
    jugador1_ev REAL NOT NULL,
    jugador2_ev REAL NOT NULL,
    
    -- Edge (ventaja sobre la casa)
    jugador1_edge REAL,
    jugador2_edge REAL,
    
    -- Recomendación
    recomendacion VARCHAR(50) NOT NULL,  -- "APOSTAR a jugador1" | "APOSTAR a jugador2" | "NO APOSTAR"
    mejor_opcion VARCHAR(200),  -- Nombre del jugador recomendado o NULL
    
    -- Confianza del modelo (basada en diferencia de probabilidades)
    confianza VARCHAR(20) CHECK(confianza IN ('Alta', 'Media', 'Baja')),
    
    -- Confianza basada en conocimiento de jugadores (NUEVO)
    confidence_level VARCHAR(20) CHECK(confidence_level IN ('HIGH', 'MEDIUM', 'LOW', 'UNKNOWN')),
    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
    player1_known BOOLEAN DEFAULT 0,  -- 1 si jugador1 está en datos históricos
    player2_known BOOLEAN DEFAULT 0,  -- 1 si jugador2 está en datos históricos
    
    -- Kelly stake recomendado
    kelly_stake_jugador1 REAL,
    kelly_stake_jugador2 REAL,
    
    -- Metadata
    modelo_version VARCHAR(50) DEFAULT '2.0.0',
    
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    UNIQUE(match_id, version)
);

CREATE INDEX idx_predictions_match ON predictions(match_id);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);


-- Tabla de apuestas registradas
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    prediction_id INTEGER NOT NULL,  -- Qué versión de predicción usamos
    
    -- Detalles de la apuesta
    jugador_apostado VARCHAR(200) NOT NULL,  -- A quién apostamos
    cuota_apostada REAL NOT NULL,
    stake REAL NOT NULL,  -- Cantidad apostada
    
    -- Timestamp de cuando registramos la apuesta
    timestamp_apuesta TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Resultado de la apuesta (NULL si el partido no ha terminado)
    resultado VARCHAR(20) CHECK(resultado IN ('ganada', 'perdida', 'cancelada', NULL)),
    ganancia REAL,  -- Ganancia neta (puede ser negativa)
    roi REAL,  -- Return on Investment
    
    -- Estado
    estado VARCHAR(20) NOT NULL DEFAULT 'activa' CHECK(estado IN ('activa', 'completada', 'cancelada')),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
    UNIQUE(match_id)  -- Solo una apuesta activa por partido
);

CREATE INDEX idx_bets_match ON bets(match_id);
CREATE INDEX idx_bets_estado ON bets(estado);
CREATE INDEX idx_bets_resultado ON bets(resultado);
CREATE INDEX idx_bets_timestamp ON bets(timestamp_apuesta);


-- Tabla de historial de cuotas (para análisis y mostrar top 3 en frontend)
CREATE TABLE IF NOT EXISTS odds_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    
    -- Cuotas
    jugador1_cuota REAL NOT NULL,
    jugador2_cuota REAL NOT NULL,
    
    -- Fuente
    bookmaker VARCHAR(100),  -- Ej: "bet365", "pinnacle"
    is_best BOOLEAN DEFAULT 0,  -- 1 si es la mejor cuota disponible
    
    -- Timestamp
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
);

CREATE INDEX idx_odds_match ON odds_history(match_id);
CREATE INDEX idx_odds_timestamp ON odds_history(timestamp);
CREATE INDEX idx_odds_bookmaker ON odds_history(bookmaker);


-- ============================================================
-- VISTAS ÚTILES
-- ============================================================

-- Vista de partidos con última predicción
CREATE VIEW IF NOT EXISTS matches_with_latest_prediction AS
SELECT 
    m.*,
    p.version as prediction_version,
    p.timestamp as prediction_timestamp,
    p.jugador1_cuota,
    p.jugador2_cuota,
    p.jugador1_probabilidad,
    p.jugador2_probabilidad,
    p.jugador1_ev,
    p.jugador2_ev,
    p.recomendacion,
    p.mejor_opcion,
    p.confianza,
    b.id as bet_id,
    b.jugador_apostado,
    b.cuota_apostada,
    b.stake,
    b.resultado as bet_resultado,
    b.ganancia
FROM matches m
LEFT JOIN predictions p ON m.id = p.match_id 
    AND p.version = (
        SELECT MAX(version) 
        FROM predictions 
        WHERE match_id = m.id
    )
LEFT JOIN bets b ON m.id = b.match_id AND b.estado = 'activa';


-- Vista de estadísticas diarias
CREATE VIEW IF NOT EXISTS daily_stats AS
SELECT 
    DATE(b.timestamp_apuesta) as fecha,
    COUNT(*) as total_apuestas,
    SUM(CASE WHEN b.resultado = 'ganada' THEN 1 ELSE 0 END) as ganadas,
    SUM(CASE WHEN b.resultado = 'perdida' THEN 1 ELSE 0 END) as perdidas,
    ROUND(CAST(SUM(CASE WHEN b.resultado = 'ganada' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*), 3) as win_rate,
    SUM(b.stake) as stake_total,
    SUM(b.ganancia) as ganancia_total,
    ROUND(SUM(b.ganancia) / SUM(b.stake), 3) as roi
FROM bets b
WHERE b.estado = 'completada'
GROUP BY DATE(b.timestamp_apuesta)
ORDER BY fecha DESC;


-- ============================================================
-- TRIGGERS para mantener updated_at
-- ============================================================

CREATE TRIGGER IF NOT EXISTS update_matches_timestamp 
AFTER UPDATE ON matches
BEGIN
    UPDATE matches SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_bets_timestamp 
AFTER UPDATE ON bets
BEGIN
    UPDATE bets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;


-- ============================================================
-- DATOS DE EJEMPLO (para testing)
-- ============================================================

-- Insertar partido de ejemplo
INSERT OR IGNORE INTO matches (
    fecha_partido, hora_inicio, torneo, ronda, superficie,
    jugador1_nombre, jugador1_ranking,
    jugador2_nombre, jugador2_ranking,
    estado
) VALUES (
    DATE('now'), '14:00', 'Australian Open', 'Cuartos de Final', 'Hard',
    'Alcaraz', 3,
    'Sinner', 4,
    'pendiente'
);
