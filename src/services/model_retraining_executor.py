"""
Background Task Executor para Re-entrenamiento del Modelo
=========================================================

Ejecuta tareas de re-entrenamiento en background sin bloquear la API.
"""

import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable
import json

logger = logging.getLogger(__name__)


class ModelRetrainingExecutor:
    """
    Ejecuta re-entrenamiento del modelo en background
    """

    def __init__(self, on_success_callback: Optional[Callable[[], None]] = None):
        """
        Inicializa el executor

        Args:
            on_success_callback: Función a llamar tras re-entrenamiento exitoso.
                Típicamente resetea el predictor en memoria para que la próxima
                predicción cargue el modelo nuevo desde disco.
        """
        self.is_running = False
        self.on_success_callback = on_success_callback
        self.last_execution = None
        self.last_result = None
        self.current_thread = None

        logger.info("✅ ModelRetrainingExecutor inicializado")

    def _execute_retraining(self, commit_info: Dict = None):
        """
        Ejecuta el script de actualización semanal en background

        Args:
            commit_info: Información del commit que trigger la actualización
        """
        try:
            self.is_running = True
            start_time = datetime.now()

            logger.info("=" * 70)
            logger.info("🚀 INICIANDO RE-ENTRENAMIENTO DEL MODELO")
            logger.info("=" * 70)

            if commit_info:
                logger.info(
                    f"📝 Trigger: Commit {commit_info.get('sha')} - {commit_info.get('message')}"
                )

            # Ruta al script de actualización
            script_path = Path("scripts/actualizacion_semanal.py")

            if not script_path.exists():
                logger.error(f"❌ Script no encontrado: {script_path}")
                self.last_result = {
                    "success": False,
                    "error": "Script de actualización no encontrado",
                    "timestamp": datetime.now().isoformat(),
                }
                return

            # Ejecutar script
            logger.info(f"🔧 Ejecutando: python {script_path}")

            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 horas máximo
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if result.returncode == 0:
                logger.info("=" * 70)
                logger.info("✅ RE-ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
                logger.info(f"⏱️  Duración: {duration:.0f} segundos ({duration/60:.1f} minutos)")
                logger.info("=" * 70)

                # Resetear predictor para que la próxima predicción cargue el modelo nuevo
                if self.on_success_callback:
                    try:
                        self.on_success_callback()
                        logger.info("🔄 Predictor reseteado - modelo nuevo se cargará en próxima predicción")
                    except Exception as cb_err:
                        logger.warning(f"⚠️ Callback post-reentrenamiento falló (no crítico): {cb_err}")

                self.last_result = {
                    "success": True,
                    "duration_seconds": duration,
                    "timestamp": end_time.isoformat(),
                    "commit": commit_info,
                }
            else:
                logger.error("=" * 70)
                logger.error("❌ ERROR EN RE-ENTRENAMIENTO")
                logger.error(f"Exit code: {result.returncode}")
                logger.error(f"Error: {result.stderr[-500:]}")  # Últimos 500 caracteres
                logger.error("=" * 70)

                self.last_result = {
                    "success": False,
                    "error": result.stderr[-500:],
                    "exit_code": result.returncode,
                    "timestamp": end_time.isoformat(),
                }

        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout: Re-entrenamiento tomó más de 2 horas")
            self.last_result = {
                "success": False,
                "error": "Timeout - proceso tomó más de 2 horas",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error ejecutando re-entrenamiento: {e}", exc_info=True)
            self.last_result = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        finally:
            self.is_running = False
            self.last_execution = datetime.now()

            # Guardar resultado en archivo
            self._save_result()

    def _save_result(self):
        """Guarda el resultado de la última ejecución"""
        try:
            result_file = Path("logs/last_retraining_result.json")
            result_file.parent.mkdir(exist_ok=True)

            with open(result_file, "w") as f:
                json.dump(self.last_result, f, indent=2)

            logger.info(f"💾 Resultado guardado en {result_file}")

        except Exception as e:
            logger.error(f"❌ Error guardando resultado: {e}")

    def start_retraining(self, commit_info: Dict = None) -> Dict:
        """
        Inicia re-entrenamiento en background

        Args:
            commit_info: Información del commit

        Returns:
            Estado de la ejecución
        """
        if self.is_running:
            logger.warning("⚠️  Re-entrenamiento ya en progreso")
            return {
                "success": False,
                "mensaje": "Re-entrenamiento ya en progreso",
                "started_at": self.last_execution.isoformat() if self.last_execution else None,
            }

        # Iniciar en thread separado
        self.current_thread = threading.Thread(
            target=self._execute_retraining, args=(commit_info,), daemon=True
        )
        self.current_thread.start()

        logger.info("🚀 Re-entrenamiento iniciado en background")

        return {
            "success": True,
            "mensaje": "Re-entrenamiento iniciado en background",
            "started_at": datetime.now().isoformat(),
        }

    def get_status(self) -> Dict:
        """
        Obtiene el estado del re-entrenamiento

        Returns:
            Estado actual
        """
        return {
            "is_running": self.is_running,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_result": self.last_result,
        }


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    executor = ModelRetrainingExecutor()

    # Simular commit info
    commit_info = {"sha": "abc123", "message": "Update ATP matches 2026", "author": "Jeff Sackmann"}

    # Iniciar re-entrenamiento
    result = executor.start_retraining(commit_info)
    print(f"\n{result}")

    # Esperar un momento y verificar estado
    import time

    time.sleep(2)

    status = executor.get_status()
    print(f"\nEstado: {status}")
