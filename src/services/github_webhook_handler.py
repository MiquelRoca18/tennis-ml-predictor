"""
GitHub Webhook Handler
======================

Maneja webhooks de GitHub para detectar commits nuevos en el repositorio
TML-Database y trigger re-entrenamiento automático del modelo.
"""

import hmac
import hashlib
import logging
from typing import Dict, Optional
from fastapi import Request, HTTPException
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GitHubWebhookHandler:
    """
    Maneja webhooks de GitHub para auto-actualización del modelo
    """
    
    def __init__(self, secret: str = None):
        """
        Args:
            secret: Secret para verificar firma del webhook
        """
        self.secret = secret or os.getenv("GITHUB_WEBHOOK_SECRET", "")
        if not self.secret:
            logger.warning("⚠️  GITHUB_WEBHOOK_SECRET no configurado - webhooks sin verificación")
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verifica la firma del webhook de GitHub
        
        Args:
            payload: Payload del webhook
            signature: Firma en header X-Hub-Signature-256
        
        Returns:
            True si la firma es válida
        """
        if not self.secret:
            logger.warning("⚠️  Webhook sin verificación de firma")
            return True  # Permitir si no hay secret configurado
        
        try:
            # GitHub envía: sha256=<hash>
            if not signature.startswith('sha256='):
                return False
            
            expected_signature = signature.split('=')[1]
            
            # Calcular HMAC
            mac = hmac.new(
                self.secret.encode('utf-8'),
                msg=payload,
                digestmod=hashlib.sha256
            )
            calculated_signature = mac.hexdigest()
            
            # Comparación segura
            return hmac.compare_digest(calculated_signature, expected_signature)
        
        except Exception as e:
            logger.error(f"❌ Error verificando firma: {e}")
            return False
    
    def should_trigger_update(self, event_data: Dict) -> bool:
        """
        Determina si el commit debe trigger una actualización
        
        Args:
            event_data: Datos del evento de GitHub
        
        Returns:
            True si debe actualizar
        """
        try:
            # Verificar que es un push event
            if 'commits' not in event_data:
                logger.info("ℹ️  No es un push event - ignorando")
                return False
            
            # Verificar que hay commits
            commits = event_data.get('commits', [])
            if not commits:
                logger.info("ℹ️  No hay commits - ignorando")
                return False
            
            # Verificar que se modificaron archivos CSV
            archivos_modificados = set()
            for commit in commits:
                archivos_modificados.update(commit.get('added', []))
                archivos_modificados.update(commit.get('modified', []))
            
            # Buscar archivos .csv
            archivos_csv = [f for f in archivos_modificados if f.endswith('.csv')]
            
            if archivos_csv:
                logger.info(f"✅ Archivos CSV modificados: {archivos_csv}")
                return True
            else:
                logger.info("ℹ️  No se modificaron archivos CSV - ignorando")
                return False
        
        except Exception as e:
            logger.error(f"❌ Error procesando evento: {e}")
            return False
    
    def extract_commit_info(self, event_data: Dict) -> Dict:
        """
        Extrae información relevante del commit
        
        Returns:
            Dict con info del commit
        """
        try:
            commits = event_data.get('commits', [])
            if not commits:
                return {}
            
            latest_commit = commits[-1]  # Último commit
            
            return {
                "sha": latest_commit.get('id', '')[:7],
                "message": latest_commit.get('message', ''),
                "author": latest_commit.get('author', {}).get('name', ''),
                "timestamp": latest_commit.get('timestamp', ''),
                "url": latest_commit.get('url', '')
            }
        
        except Exception as e:
            logger.error(f"❌ Error extrayendo info del commit: {e}")
            return {}


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    handler = GitHubWebhookHandler()
    
    # Ejemplo de evento de GitHub (TML-Database)
    event_data = {
        "commits": [
            {
                "id": "abc123def456",
                "message": "Update ATP matches 2026",
                "author": {"name": "TML Database"},
                "timestamp": "2026-01-08T20:00:00Z",
                "url": "https://github.com/Tennismylife/TML-Database/commit/abc123",
                "added": [],
                "modified": ["2026.csv"],
                "removed": []
            }
        ]
    }
    
    # Verificar si debe actualizar
    should_update = handler.should_trigger_update(event_data)
    print(f"¿Debe actualizar? {should_update}")
    
    # Extraer info
    if should_update:
        commit_info = handler.extract_commit_info(event_data)
        print(f"\nInfo del commit:")
        print(f"  SHA: {commit_info['sha']}")
        print(f"  Mensaje: {commit_info['message']}")
        print(f"  Autor: {commit_info['author']}")
