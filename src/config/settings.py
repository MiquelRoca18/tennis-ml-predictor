"""
Configuración Centralizada - Tennis ML Predictor
===============================================

Módulo de configuración unificado que consolida toda la configuración
del proyecto en un solo lugar.

Uso:
    from src.config.settings import Config

    # Acceder a configuración
    api_key = Config.ODDS_API_KEY
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

    # ==================== DATOS ====================
    DATA_PATH = os.getenv("DATA_PATH", "datos/processed/dataset_final.csv")
    DATA_BACKUP_DIR = os.getenv("DATA_BACKUP_DIR", "backups")

    # ==================== PARÁMETROS DE APUESTA ====================
    # CONFIGURACIÓN CONSERVADORA (alineada con backtesting exitoso)
    EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.10"))  # 10% EV mínimo (backtesting validado)
    EV_THRESHOLD_ALERT = float(os.getenv("EV_THRESHOLD_ALERT", "0.15"))  # 15% para alertas
    MAX_CUOTA = float(os.getenv("MAX_CUOTA", "2.0"))  # Solo favoritos (cuota < 2.0)
    MIN_PROBABILIDAD = float(os.getenv("MIN_PROBABILIDAD", "0.60"))  # Solo cuando modelo > 60%
    BANKROLL_INICIAL = float(os.getenv("BANKROLL_INICIAL", "1000"))

    # ==================== KELLY CRITERION ====================
    KELLY_ENABLED = os.getenv("KELLY_ENABLED", "true").lower() == "true"
    KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.05"))  # 5% Kelly (muy conservador)

    # ==================== BOOKMAKERS ====================
    # The Odds API
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

        # Validaciones críticas (siempre requeridas)
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Modelo no encontrado: {cls.MODEL_PATH}")

        # Validaciones opcionales (solo en modo strict)
        if strict:
            if not cls.ODDS_API_KEY:
                errors.append("ODDS_API_KEY no configurada")

            if cls.EMAIL_ENABLED and (not cls.EMAIL_ADDRESS or not cls.EMAIL_PASSWORD):
                errors.append("Email habilitado pero credenciales no configuradas")
        else:
            # En modo no-strict, solo advertir
            if not cls.ODDS_API_KEY:
                warnings.append(
                    "ODDS_API_KEY no configurada (funcionalidad de bookmakers limitada)"
                )

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
        if show_secrets:
            print(f"   ODDS_API_KEY: {cls.ODDS_API_KEY or '❌ No configurada'}")
        else:
            print(
                f"   ODDS_API_KEY: {'✅ Configurada' if cls.ODDS_API_KEY else '❌ No configurada'}"
            )

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

        print(f"\n🌐 Bookmakers:")
        print(f"   API Base URL: {cls.ODDS_API_BASE_URL}")
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

# API Keys
ODDS_API_KEY=tu_api_key_aqui

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

# Betting Parameters (CONSERVADOR - alineado con backtesting exitoso)
EV_THRESHOLD=0.10
EV_THRESHOLD_ALERT=0.15
MAX_CUOTA=2.0
MIN_PROBABILIDAD=0.60
BANKROLL_INICIAL=1000

# Kelly Criterion
KELLY_ENABLED=true
KELLY_FRACTION=0.05

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
