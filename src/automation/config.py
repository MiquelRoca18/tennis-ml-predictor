"""
Gestión de configuración con variables de entorno
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

class Config:
    """Configuración del sistema"""
    
    # API Keys
    ODDS_API_KEY = os.getenv('ODDS_API_KEY')
    
    # Email
    EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', 587))
    EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
    
    # Database
    DB_PATH = os.getenv('DB_PATH', 'apuestas_tracker_demo.db')
    
    # Modelo
    MODEL_PATH = os.getenv('MODEL_PATH', 'modelos/modelo_production.pkl')
    MODEL_BACKUP_DIR = os.getenv('MODEL_BACKUP_DIR', 'modelos/backups')
    
    # Datos
    DATA_PATH = os.getenv('DATA_PATH', 'datos/processed/dataset_features_completas.csv')
    DATA_BACKUP_DIR = os.getenv('DATA_BACKUP_DIR', 'backups')
    
    # Parámetros
    EV_THRESHOLD = float(os.getenv('EV_THRESHOLD', 0.03))
    BANKROLL_INICIAL = float(os.getenv('BANKROLL_INICIAL', 1000))
    
    # Reentrenamiento
    RETRAIN_STRATEGY = os.getenv('RETRAIN_STRATEGY', 'semanal')  # diario, semanal, mensual, threshold
    RETRAIN_THRESHOLD_MATCHES = int(os.getenv('RETRAIN_THRESHOLD_MATCHES', 100))
    
    # Logging
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate(cls):
        """Valida que las configuraciones críticas estén presentes"""
        errors = []
        
        if not cls.ODDS_API_KEY:
            errors.append("ODDS_API_KEY no configurada")
        
        if not cls.EMAIL_ADDRESS or not cls.EMAIL_PASSWORD:
            errors.append("Credenciales de email no configuradas")
        
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Modelo no encontrado: {cls.MODEL_PATH}")
        
        if errors:
            raise ValueError(f"Errores de configuración:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Crea directorios necesarios si no existen"""
        Path(cls.LOG_DIR).mkdir(exist_ok=True)
        Path(cls.DATA_BACKUP_DIR).mkdir(exist_ok=True)
        Path(cls.MODEL_BACKUP_DIR).mkdir(exist_ok=True)


# Template para .env.example
ENV_EXAMPLE = """# API Keys
ODDS_API_KEY=tu_api_key_aqui

# Email Configuration
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=tu@email.com
EMAIL_PASSWORD=tu_app_password

# Database
DB_PATH=apuestas_tracker_demo.db

# Model
MODEL_PATH=modelos/modelo_production.pkl
MODEL_BACKUP_DIR=modelos/backups

# Data
DATA_PATH=datos/processed/dataset_features_completas.csv
DATA_BACKUP_DIR=backups

# Parameters
EV_THRESHOLD=0.03
BANKROLL_INICIAL=1000

# Retraining
RETRAIN_STRATEGY=semanal
RETRAIN_THRESHOLD_MATCHES=100

# Logging
LOG_DIR=logs
LOG_LEVEL=INFO
"""

if __name__ == "__main__":
    # Crear .env.example
    with open('.env.example', 'w') as f:
        f.write(ENV_EXAMPLE)
    print("✅ .env.example creado")
    
    # Validar configuración actual
    try:
        Config.validate()
        print("✅ Configuración válida")
    except ValueError as e:
        print(f"⚠️  {e}")
