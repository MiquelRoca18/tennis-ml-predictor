"""
Configuración Centralizada - Tennis ML Predictor
===============================================

Módulo de configuración unificado que consolida toda la configuración
del proyecto en un solo lugar.

Uso:
    from src.config.settings import Config

    # Acceder a configuración
    api_key = Config.API_TENNIS_API_KEY  # o Config.MODEL_PATH, etc.
    model_path = Config.MODEL_PATH

    # Validar configuración
    Config.validate()
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Config:
    """
    Configuración centralizada del sistema

    Todas las configuraciones se cargan desde variables de entorno (.env)
    con valores por defecto razonables.
    """

    # ==================== API KEYS ====================
    # API-Tennis (obligatoria para partidos y cuotas). La lee api_tennis_client por os.getenv.
    API_TENNIS_API_KEY = os.getenv("API_TENNIS_API_KEY", "")
    # Legacy: The Odds API no se usa; las cuotas vienen de API-Tennis. Se mantiene por compatibilidad.
    ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

    # ==================== EMAIL ====================
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")

    # ==================== DATABASE ====================
    # PostgreSQL (Railway) - detectado automáticamente por DATABASE_URL
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    
    # SQLite fallback (local development)
    DB_PATH = os.getenv("DB_PATH", "apuestas_tracker.db")
    MATCHES_DB_PATH = os.getenv("MATCHES_DB_PATH", "matches_v2.db")
    
    @classmethod
    def is_postgres(cls) -> bool:
        """Detecta si estamos usando PostgreSQL"""
        return bool(cls.DATABASE_URL)

    # ==================== MODELO ====================
    MODEL_PATH = os.getenv("MODEL_PATH", "modelos/random_forest_calibrado.pkl")
    MODEL_BACKUP_DIR = os.getenv("MODEL_BACKUP_DIR", "modelos/archive")

    # Baseline ELO (sin ML): prob = BASELINE_ELO_PESO * prob_elo + (1 - BASELINE_ELO_PESO) * prob_mercado
    USE_BASELINE_ELO = os.getenv("USE_BASELINE_ELO", "false").lower() == "true"
    BASELINE_ELO_PESO = float(os.getenv("BASELINE_ELO_PESO", "0.6"))  # 60% ELO, 40% mercado

    # ==================== DATOS ====================
    DATA_PATH = os.getenv("DATA_PATH", "datos/processed/dataset_final.csv")
    DATA_BACKUP_DIR = os.getenv("DATA_BACKUP_DIR", "backups")

    # ==================== PARÁMETROS DE APUESTA ====================
    # Valores por defecto = CONFIG_MEJOR del backtesting (~105 apuestas/año, ROI ~113% 2021-2024).
    # Para igualar producción al backtesting: EV_THRESHOLD=0.03, MIN_PROBABILIDAD=0.50, MAX_CUOTA=3.0
    EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.03"))  # 3% EV mínimo (CONFIG_MEJOR)
    EV_THRESHOLD_ALERT = float(os.getenv("EV_THRESHOLD_ALERT", "0.15"))  # 15% para alertas
    MAX_CUOTA = float(os.getenv("MAX_CUOTA", "3.0"))  # Cuota máxima recomendada (CONFIG_MEJOR)
    MIN_PROBABILIDAD = float(os.getenv("MIN_PROBABILIDAD", "0.50"))  # Prob mínima 50% (CONFIG_MEJOR)
    BANKROLL_INICIAL = float(os.getenv("BANKROLL_INICIAL", "1000"))

    # ==================== KELLY CRITERION ====================
    KELLY_ENABLED = os.getenv("KELLY_ENABLED", "true").lower() == "true"
    KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.05"))  # 5% Kelly (muy conservador)
    MIN_STAKE_EUR = float(os.getenv("MIN_STAKE_EUR", "5"))  # Mínimo 5€ (casas suelen 0.10€–1€; 5€ conservador)
    MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.10"))  # Máximo 10% del bankroll
    # Límite máximo por apuesta (€): compatible con la mayoría de casas (250k–500k ganancias máx;
    # depósito España DGOJ 600€/24h). 250€ permite 1–2 apuestas/día sin superar límite diario.
    MAX_STAKE_EUR = float(os.getenv("MAX_STAKE_EUR", "250"))

    # ==================== BOOKMAKERS (legacy) ====================
    # The Odds API: no se usa en el proyecto; partidos y cuotas vienen de API-Tennis.
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
    ODDS_REGIONS = os.getenv("ODDS_REGIONS", "eu,us")
    ODDS_MARKETS = os.getenv("ODDS_MARKETS", "h2h")
    ODDS_FORMAT = os.getenv("ODDS_FORMAT", "decimal")

    # Límites de API
    ODDS_MAX_REQUESTS_PER_MONTH = int(os.getenv("ODDS_MAX_REQUESTS_PER_MONTH", "500"))
    ODDS_WARNING_THRESHOLD = int(os.getenv("ODDS_WARNING_THRESHOLD", "50"))

    # Caché
    CACHE_DIR = Path(os.getenv("CACHE_DIR", "datos/cache_cuotas"))
    CACHE_DURATION_MINUTES = int(os.getenv("CACHE_DURATION_MINUTES", "30"))

    # ==================== AUTOMATIZACIÓN ====================
    # Reentrenamiento
    RETRAIN_STRATEGY = os.getenv(
        "RETRAIN_STRATEGY", "semanal"
    )  # diario, semanal, mensual, threshold
    RETRAIN_THRESHOLD_MATCHES = int(os.getenv("RETRAIN_THRESHOLD_MATCHES", "100"))

    # ==================== LOGGING ====================
    LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # ==================== DEPORTES ====================
    SPORT_ATP = "tennis_atp"
    SPORT_WTA = "tennis_wta"

    @classmethod
    def validate(cls, strict=False):
        """
        Valida que las configuraciones críticas estén presentes

        Args:
            strict: Si True, requiere todas las configuraciones.
                   Si False, solo valida las críticas.

        Returns:
            tuple: (is_valid, errors_list)
        """
        errors = []
        warnings = []

        # Validaciones críticas (sistema usa solo baseline ELO + mercado; no requiere .pkl)
        # MODEL_PATH se mantiene por compatibilidad pero no se usa en runtime

        # Validaciones opcionales (solo en modo strict)
        if strict:
            if cls.EMAIL_ENABLED and (not cls.EMAIL_ADDRESS or not cls.EMAIL_PASSWORD):
                errors.append("Email habilitado pero credenciales no configuradas")
        else:
            # En modo no-strict, solo advertir (ODDS_API_KEY no se usa; cuotas vía API-Tennis)
            if cls.EMAIL_ENABLED and (not cls.EMAIL_ADDRESS or not cls.EMAIL_PASSWORD):
                warnings.append("Email habilitado pero credenciales no configuradas")

        is_valid = len(errors) == 0

        return is_valid, errors, warnings

    @classmethod
    def create_directories(cls):
        """Crea directorios necesarios si no existen"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        Path(cls.DATA_BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.MODEL_BACKUP_DIR).mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("✅ Directorios creados/verificados")

    @classmethod
    def print_config(cls, show_secrets=False):
        """
        Muestra la configuración actual

        Args:
            show_secrets: Si True, muestra valores de API keys y passwords
        """
        print("\n" + "=" * 60)
        print("⚙️  CONFIGURACIÓN DEL SISTEMA")
        print("=" * 60)

        print(f"\n🔑 API Keys:")
        print(f"   API_TENNIS_API_KEY: {'✅ Configurada' if cls.API_TENNIS_API_KEY else '❌ No configurada'} (partidos y cuotas)")
        if show_secrets:
            print(f"   ODDS_API_KEY (legacy, no usada): {cls.ODDS_API_KEY or '—'}")
        else:
            print(f"   ODDS_API_KEY (legacy, no usada): {'—'}")

        print(f"\n📧 Email:")
        print(f"   Habilitado: {'✅ Sí' if cls.EMAIL_ENABLED else '❌ No'}")
        if cls.EMAIL_ENABLED:
            print(f"   Servidor: {cls.EMAIL_SMTP_SERVER}:{cls.EMAIL_SMTP_PORT}")
            print(
                f"   Dirección: {cls.EMAIL_ADDRESS if cls.EMAIL_ADDRESS else '❌ No configurada'}"
            )

        print(f"\n💾 Base de Datos:")
        print(f"   Tipo: {'PostgreSQL (Railway)' if cls.is_postgres() else 'SQLite (Local)'}")
        if cls.is_postgres():
            # Mostrar solo host, no credenciales
            db_url = cls.DATABASE_URL
            if "@" in db_url:
                host_part = db_url.split("@")[-1].split("/")[0]
                print(f"   Host: {host_part}")
        else:
            print(f"   DB Path: {cls.DB_PATH}")
            print(f"   Matches DB: {cls.MATCHES_DB_PATH}")

        print(f"\n🤖 Modelo:")
        print(f"   Path: {cls.MODEL_PATH}")
        print(f"   Existe: {'✅ Sí' if Path(cls.MODEL_PATH).exists() else '❌ No'}")
        print(f"   Backup Dir: {cls.MODEL_BACKUP_DIR}")

        print(f"\n📊 Datos:")
        print(f"   Path: {cls.DATA_PATH}")
        print(f"   Backup Dir: {cls.DATA_BACKUP_DIR}")

        print(f"\n💰 Parámetros de Apuesta:")
        print(f"   EV Threshold: {cls.EV_THRESHOLD*100:.1f}%")
        print(f"   EV Alert: {cls.EV_THRESHOLD_ALERT*100:.1f}%")
        print(f"   Bankroll Inicial: {cls.BANKROLL_INICIAL}€")

        print(f"\n💎 Kelly Criterion:")
        print(f"   Habilitado: {'✅ Sí' if cls.KELLY_ENABLED else '❌ No'}")
        if cls.KELLY_ENABLED:
            print(f"   Fracción: {cls.KELLY_FRACTION*100:.0f}%")

        print(f"\n🌐 Bookmakers (legacy, no usados; cuotas vía API-Tennis):")
        print(f"   ODDS_API_BASE_URL: {cls.ODDS_API_BASE_URL}")
        print(f"   Regiones: {cls.ODDS_REGIONS}")
        print(f"   Formato: {cls.ODDS_FORMAT}")
        print(f"   Max Requests/mes: {cls.ODDS_MAX_REQUESTS_PER_MONTH}")

        print(f"\n💾 Caché:")
        print(f"   Directorio: {cls.CACHE_DIR}")
        print(f"   Duración: {cls.CACHE_DURATION_MINUTES} minutos")

        print(f"\n🤖 Automatización:")
        print(f"   Estrategia Reentrenamiento: {cls.RETRAIN_STRATEGY}")
        print(f"   Threshold Matches: {cls.RETRAIN_THRESHOLD_MATCHES}")

        print(f"\n📝 Logging:")
        print(f"   Directorio: {cls.LOG_DIR}")
        print(f"   Nivel: {cls.LOG_LEVEL}")

        # Validar
        is_valid, errors, warnings = cls.validate(strict=False)

        print("\n" + "=" * 60)
        if is_valid:
            print("✅ CONFIGURACIÓN VÁLIDA")
        else:
            print("❌ ERRORES EN CONFIGURACIÓN:")
            for error in errors:
                print(f"   - {error}")

        if warnings:
            print("\n⚠️  ADVERTENCIAS:")
            for warning in warnings:
                print(f"   - {warning}")

        print("=" * 60)

        return is_valid


# Template para .env
ENV_TEMPLATE = """# ===========================================
# Tennis ML Predictor - Configuración
# ===========================================

# API-Tennis (obligatoria para partidos y cuotas)
API_TENNIS_API_KEY=tu_api_key_api_tennis

# ODDS_API_KEY: no se usa; las cuotas vienen de API-Tennis
# ODDS_API_KEY=opcional_no_usada

# Email Configuration (opcional)
EMAIL_ENABLED=false
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=tu@email.com
EMAIL_PASSWORD=tu_app_password
EMAIL_RECIPIENT=tu@email.com

# Database
DB_PATH=apuestas_tracker.db

# Model
MODEL_PATH=modelos/production/random_forest_calibrado.pkl
MODEL_BACKUP_DIR=modelos/archive

# Data
DATA_PATH=datos/processed/dataset_final.csv
DATA_BACKUP_DIR=backups

# Betting Parameters - CONFIG_MEJOR del backtesting (~105 apuestas/año, ROI ~113%)
EV_THRESHOLD=0.03
EV_THRESHOLD_ALERT=0.15
MAX_CUOTA=3.0
MIN_PROBABILIDAD=0.50
BANKROLL_INICIAL=1000

# Kelly Criterion (igual que backtesting)
KELLY_ENABLED=true
KELLY_FRACTION=0.05
MIN_STAKE_EUR=5
MAX_STAKE_PCT=0.10
# Límite por apuesta (€): compatible con casas 250k–2M€ y depósito DGOJ 600€/24h
MAX_STAKE_EUR=250

# Bookmakers
ODDS_REGIONS=eu,us
ODDS_MARKETS=h2h
ODDS_FORMAT=decimal
ODDS_MAX_REQUESTS_PER_MONTH=500
ODDS_WARNING_THRESHOLD=50

# Cache
CACHE_DIR=datos/cache_cuotas
CACHE_DURATION_MINUTES=30

# Automation
RETRAIN_STRATEGY=semanal
RETRAIN_THRESHOLD_MATCHES=100

# Logging
LOG_DIR=logs
LOG_LEVEL=INFO
"""


if __name__ == "__main__":
    print("🎾 Tennis ML Predictor - Configuración Centralizada\n")

    # Mostrar configuración
    is_valid = Config.print_config(show_secrets=False)

    # Crear directorios
    print("\n📁 Creando directorios...")
    Config.create_directories()

    # Ofrecer crear .env.template
    if not is_valid:
        print("\n💡 Tip: Copia .env.template a .env y configura tus valores")

        create_template = input("\n¿Crear .env.template? (s/n): ").lower()
        if create_template == "s":
            with open(".env.template", "w") as f:
                f.write(ENV_TEMPLATE)
            print("✅ .env.template creado")
