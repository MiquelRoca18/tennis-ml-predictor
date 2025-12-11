"""
Configuración centralizada para el módulo de bookmakers

Gestiona API keys, configuración de email, parámetros de caché y umbrales.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class BookmakerConfig:
    """
    Configuración centralizada para el sistema de bookmakers
    """
    
    # The Odds API
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Deportes disponibles
    SPORT_ATP = 'tennis_atp'
    SPORT_WTA = 'tennis_wta'
    
    # Regiones de bookmakers
    REGIONS = 'eu,us'  # Europa y USA
    
    # Mercados
    MARKETS = 'h2h'  # Head-to-head (ganador del partido)
    
    # Formato de cuotas
    ODDS_FORMAT = 'decimal'
    
    # Límites de API
    MAX_REQUESTS_PER_MONTH = 500  # Plan gratuito
    WARNING_THRESHOLD = 50  # Alertar cuando queden menos de 50 requests
    
    # Caché
    CACHE_DIR = Path(__file__).parent.parent.parent / 'datos' / 'cache_cuotas'
    CACHE_DURATION_MINUTES = 30  # Duración del caché en minutos
    
    # Umbrales de EV
    EV_THRESHOLD_DEFAULT = 0.03  # 3% EV mínimo para apostar
    EV_THRESHOLD_ALERT = 0.05  # 5% EV para enviar alerta
    
    # Configuración de Email (opcional)
    EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')  # App Password
    EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', '')
    
    @classmethod
    def validate_config(cls):
        """
        Valida que la configuración esté completa
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not cls.ODDS_API_KEY:
            return False, "⚠️  ODDS_API_KEY no configurada. Define la variable de entorno ODDS_API_KEY"
        
        if cls.EMAIL_ENABLED:
            if not cls.EMAIL_ADDRESS or not cls.EMAIL_PASSWORD:
                return False, "⚠️  Email habilitado pero EMAIL_ADDRESS o EMAIL_PASSWORD no configurados"
        
        return True, "✅ Configuración válida"
    
    @classmethod
    def print_config(cls):
        """
        Muestra la configuración actual (sin mostrar secrets)
        """
        print("\n" + "="*60)
        print("⚙️  CONFIGURACIÓN DE BOOKMAKERS")
        print("="*60)
        
        print(f"\n🔑 API Configuration:")
        print(f"   API Key: {'✅ Configurada' if cls.ODDS_API_KEY else '❌ No configurada'}")
        print(f"   Base URL: {cls.ODDS_API_BASE_URL}")
        print(f"   Regiones: {cls.REGIONS}")
        print(f"   Formato: {cls.ODDS_FORMAT}")
        
        print(f"\n📊 Límites:")
        print(f"   Max requests/mes: {cls.MAX_REQUESTS_PER_MONTH}")
        print(f"   Umbral de alerta: {cls.WARNING_THRESHOLD} requests")
        
        print(f"\n💾 Caché:")
        print(f"   Directorio: {cls.CACHE_DIR}")
        print(f"   Duración: {cls.CACHE_DURATION_MINUTES} minutos")
        
        print(f"\n📈 Umbrales de EV:")
        print(f"   EV mínimo: {cls.EV_THRESHOLD_DEFAULT*100:.1f}%")
        print(f"   EV para alerta: {cls.EV_THRESHOLD_ALERT*100:.1f}%")
        
        print(f"\n📧 Email:")
        print(f"   Habilitado: {'✅ Sí' if cls.EMAIL_ENABLED else '❌ No'}")
        if cls.EMAIL_ENABLED:
            print(f"   Servidor: {cls.SMTP_SERVER}:{cls.SMTP_PORT}")
            print(f"   Dirección: {cls.EMAIL_ADDRESS if cls.EMAIL_ADDRESS else '❌ No configurada'}")
        
        # Validar
        is_valid, message = cls.validate_config()
        print(f"\n{message}")
        print("="*60)
        
        return is_valid


# Ejemplo de uso
if __name__ == "__main__":
    config = BookmakerConfig()
    is_valid = config.print_config()
    
    if not is_valid:
        print("\n⚠️  Por favor, configura las variables de entorno necesarias")
        print("\nEjemplo de archivo .env:")
        print("-" * 40)
        print("ODDS_API_KEY=tu_api_key_aqui")
        print("EMAIL_ENABLED=true")
        print("EMAIL_ADDRESS=tu@email.com")
        print("EMAIL_PASSWORD=tu_app_password")
        print("EMAIL_RECIPIENT=tu@email.com")
        print("-" * 40)
