"""
GitHub Polling Monitor
=====================

Monitorea el repositorio TML-Database para detectar commits nuevos
sin necesidad de webhooks (no requiere acceso admin al repo).

En Railway: usa PostgreSQL para persistir el último SHA procesado (sobrevive restarts).
Local: usa archivo logs/last_commit_sha.json.
"""

import requests
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.database.match_database import MatchDatabase

logger = logging.getLogger(__name__)


class GitHubPollingMonitor:
    """
    Monitorea repositorio GitHub mediante polling (sin webhooks)
    """

    def __init__(
        self,
        repo_owner="Tennismylife",
        repo_name="TML-Database",
        db: Optional["MatchDatabase"] = None,
    ):
        """
        Args:
            repo_owner: Dueño del repositorio
            repo_name: Nombre del repositorio
            db: MatchDatabase opcional. Si es PostgreSQL, persiste SHA en DB (sobrevive restarts en Railway)
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
        self.state_file = Path("logs/last_commit_sha.json")
        self.db = db
        self._use_db = db is not None and getattr(db, "is_postgres", False)

        logger.info(f"✅ GitHubPollingMonitor inicializado: {repo_owner}/{repo_name}")
        if self._use_db:
            logger.info("   Persistencia: PostgreSQL (sobrevive restarts)")

    def get_latest_commit(self) -> Optional[Dict]:
        """
        Obtiene el último commit del repositorio

        Returns:
            Dict con información del commit o None si hay error
        """
        try:
            logger.info(f"📡 Consultando último commit de {self.repo_owner}/{self.repo_name}...")

            # GitHub API - obtener último commit
            response = requests.get(
                self.api_url, params={"per_page": 1}, timeout=10  # Solo el más reciente
            )
            response.raise_for_status()

            commits = response.json()
            if not commits:
                logger.warning("⚠️  No se encontraron commits")
                return None

            latest = commits[0]

            commit_info = {
                "sha": latest["sha"],
                "short_sha": latest["sha"][:7],
                "message": latest["commit"]["message"],
                "author": latest["commit"]["author"]["name"],
                "date": latest["commit"]["author"]["date"],
                "url": latest["html_url"],
            }

            logger.info(
                f"✅ Último commit: {commit_info['short_sha']} - {commit_info['message'][:50]}"
            )

            return commit_info

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error consultando GitHub API: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error procesando commit: {e}")
            return None

    def get_last_processed_sha(self) -> Optional[str]:
        """
        Obtiene el SHA del último commit procesado.
        Usa PostgreSQL si disponible (persiste entre restarts en Railway).
        """
        try:
            if self._use_db:
                return self.db.get_retraining_last_sha()

            if not self.state_file.exists():
                return None

            with open(self.state_file, "r") as f:
                data = json.load(f)
                return data.get("sha")

        except Exception as e:
            logger.error(f"❌ Error leyendo estado: {e}")
            return None

    def save_processed_sha(self, commit_info: Dict):
        """
        Guarda el SHA del commit procesado.
        Usa PostgreSQL si disponible (persiste entre restarts en Railway).
        """
        try:
            if self._use_db:
                self.db.set_retraining_last_sha(
                    commit_info["sha"],
                    metadata={
                        "short_sha": commit_info["short_sha"],
                        "message": commit_info["message"],
                        "processed_at": datetime.now().isoformat(),
                    },
                )
                logger.info(f"💾 Estado guardado en DB: {commit_info['short_sha']}")
                return

            self.state_file.parent.mkdir(exist_ok=True)

            data = {
                "sha": commit_info["sha"],
                "short_sha": commit_info["short_sha"],
                "message": commit_info["message"],
                "processed_at": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"💾 Estado guardado: {commit_info['short_sha']}")

        except Exception as e:
            logger.error(f"❌ Error guardando estado: {e}")

    def check_for_new_commits(self) -> Optional[Dict]:
        """
        Verifica si hay commits nuevos desde la última verificación

        Returns:
            Información del commit nuevo o None si no hay cambios
        """
        try:
            # Obtener último commit del repo
            latest_commit = self.get_latest_commit()
            if not latest_commit:
                return None

            # Obtener último commit procesado
            last_processed_sha = self.get_last_processed_sha()

            # Primera ejecución
            if not last_processed_sha:
                logger.info("ℹ️  Primera ejecución - guardando estado actual")
                self.save_processed_sha(latest_commit)
                return None

            # Comparar SHAs
            if latest_commit["sha"] == last_processed_sha:
                logger.info("ℹ️  No hay commits nuevos")
                return None

            # ¡Hay commit nuevo!
            logger.info(f"🆕 ¡Commit nuevo detectado!")
            logger.info(f"   Anterior: {last_processed_sha[:7]}")
            logger.info(f"   Nuevo: {latest_commit['short_sha']}")
            logger.info(f"   Mensaje: {latest_commit['message']}")

            # Guardar nuevo estado
            self.save_processed_sha(latest_commit)

            return latest_commit

        except Exception as e:
            logger.error(f"❌ Error verificando commits: {e}")
            return None


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Crear monitor
    monitor = GitHubPollingMonitor()

    # Verificar commits nuevos
    new_commit = monitor.check_for_new_commits()

    if new_commit:
        print(f"\n🎉 ¡Commit nuevo detectado!")
        print(f"SHA: {new_commit['sha']}")
        print(f"Mensaje: {new_commit['message']}")
        print(f"Autor: {new_commit['author']}")
        print(f"Fecha: {new_commit['date']}")
        print(f"\n→ Aquí se ejecutaría el re-entrenamiento del modelo")
    else:
        print("\nℹ️  No hay commits nuevos")
