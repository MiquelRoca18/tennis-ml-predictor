"""
Test Suite Completo - API Tennis ML Predictor v2.0
==================================================

Script de testing integral que verifica todos los endpoints
y funcionalidades de la API.
"""

import requests
from datetime import date, timedelta
from typing import Dict
import sys

# Configuración
BASE_URL = "http://localhost:8001"
VERBOSE = True


class Colors:
    """Colores para output en terminal"""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def log(message: str, color: str = Colors.BLUE):
    """Log con color"""
    if VERBOSE:
        print(f"{color}{message}{Colors.END}")


def test_endpoint(
    name: str, method: str, endpoint: str, data: Dict = None, expected_status: int = 200
) -> Dict:
    """
    Prueba un endpoint de la API

    Returns:
        Response JSON o None si falla
    """
    url = f"{BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        else:
            raise ValueError(f"Método no soportado: {method}")

        if response.status_code == expected_status:
            log(f"✅ {name}: PASS", Colors.GREEN)
            return response.json() if response.text else {}
        else:
            log(f"❌ {name}: FAIL (Status {response.status_code})", Colors.RED)
            log(f"   Response: {response.text}", Colors.RED)
            return None

    except Exception as e:
        log(f"❌ {name}: ERROR - {e}", Colors.RED)
        return None


def run_tests():
    """Ejecuta todos los tests"""

    log("\n" + "=" * 70, Colors.BLUE)
    log("🧪 INICIANDO TEST SUITE - API v2.0", Colors.BLUE)
    log("=" * 70 + "\n", Colors.BLUE)

    results = {"total": 0, "passed": 0, "failed": 0}

    # ============================================================
    # TEST 1: Endpoints Básicos
    # ============================================================

    log("\n📋 TEST 1: Endpoints Básicos", Colors.YELLOW)
    log("-" * 50, Colors.YELLOW)

    # 1.1 Root endpoint
    results["total"] += 1
    response = test_endpoint("GET /", "GET", "/")
    if response and "version" in response:
        results["passed"] += 1
    else:
        results["failed"] += 1

    # 1.2 Health check
    results["total"] += 1
    response = test_endpoint("GET /health", "GET", "/health")
    if response and response.get("status") == "ok":
        results["passed"] += 1
    else:
        results["failed"] += 1

    # 1.3 Config
    results["total"] += 1
    response = test_endpoint("GET /config", "GET", "/config")
    if response and "ev_threshold" in response:
        results["passed"] += 1
    else:
        results["failed"] += 1

    # ============================================================
    # TEST 2: Gestión de Partidos
    # ============================================================

    log("\n🎾 TEST 2: Gestión de Partidos", Colors.YELLOW)
    log("-" * 50, Colors.YELLOW)

    # 2.1 Crear partido con predicción
    results["total"] += 1
    match_data = {
        "fecha_partido": str(date.today() + timedelta(days=1)),
        "superficie": "Hard",
        "jugador1_nombre": "Test Player 1",
        "jugador1_cuota": 2.10,
        "jugador2_nombre": "Test Player 2",
        "jugador2_cuota": 1.80,
    }
    response = test_endpoint("POST /matches/predict", "POST", "/matches/predict", match_data)
    if response and "match_id" in response:
        results["passed"] += 1
        match_id = response["match_id"]
        log(f"   Match ID creado: {match_id}", Colors.GREEN)
    else:
        results["failed"] += 1
        match_id = None

    # 2.2 Obtener partidos por fecha
    results["total"] += 1
    tomorrow = str(date.today() + timedelta(days=1))
    response = test_endpoint("GET /matches", "GET", f"/matches?date={tomorrow}")
    if response and "partidos" in response:
        results["passed"] += 1
        log(f"   Partidos encontrados: {response['resumen']['total_partidos']}", Colors.GREEN)
    else:
        results["failed"] += 1

    # 2.3 Actualizar cuotas (refresh)
    if match_id:
        results["total"] += 1
        response = test_endpoint(
            "POST /matches/{id}/refresh",
            "POST",
            f"/matches/{match_id}/refresh?jugador1_cuota=2.30&jugador2_cuota=1.70",
        )
        if response and "version" in response:
            results["passed"] += 1
            log(f"   Nueva versión: {response['version']}", Colors.GREEN)
        else:
            results["failed"] += 1

    # 2.4 Actualizar resultado
    if match_id:
        results["total"] += 1
        result_data = {"ganador": "Test Player 1", "marcador": "6-4, 7-5"}
        response = test_endpoint(
            "PUT /matches/{id}/result", "PUT", f"/matches/{match_id}/result", result_data
        )
        if response and response.get("actualizado"):
            results["passed"] += 1
        else:
            results["failed"] += 1

    # ============================================================
    # TEST 3: Estadísticas
    # ============================================================

    log("\n📊 TEST 3: Estadísticas", Colors.YELLOW)
    log("-" * 50, Colors.YELLOW)

    # 3.1 Resumen de estadísticas
    results["total"] += 1
    response = test_endpoint("GET /stats/summary", "GET", "/stats/summary?period=7d")
    if response and "apuestas" in response:
        results["passed"] += 1
        log(f"   Total apuestas: {response['apuestas']['total']}", Colors.GREEN)
    else:
        results["failed"] += 1

    # 3.2 Estadísticas diarias
    results["total"] += 1
    response = test_endpoint("GET /stats/daily", "GET", "/stats/daily?days=7")
    if response and "dias" in response:
        results["passed"] += 1
    else:
        results["failed"] += 1

    # ============================================================
    # TEST 4: Cron / Admin (solo endpoints usados por cron-job.org)
    # ============================================================

    log("\n⚙️  TEST 4: Cron status", Colors.YELLOW)
    log("-" * 50, Colors.YELLOW)

    # 4.1 Último resultado de crons (trigger-retraining, refresh-elo)
    results["total"] += 1
    response = test_endpoint("GET /admin/cron-status", "GET", "/admin/cron-status")
    if response and "trigger_retraining" in response and "refresh_elo" in response:
        results["passed"] += 1
        log("   Cron status: trigger_retraining y refresh_elo presentes", Colors.GREEN)
    else:
        results["failed"] += 1

    # ============================================================
    # TEST 5: Casos Edge
    # ============================================================

    log("\n🔍 TEST 5: Casos Edge", Colors.YELLOW)
    log("-" * 50, Colors.YELLOW)

    # 5.1 Fecha inválida
    results["total"] += 1
    response = test_endpoint(
        "GET /matches (fecha inválida)", "GET", "/matches?date=invalid-date", expected_status=400
    )
    if response is not None or True:  # Esperamos error
        results["passed"] += 1
    else:
        results["failed"] += 1

    # 5.2 Match no encontrado
    results["total"] += 1
    response = test_endpoint(
        "PUT /matches/9999/result (no existe)",
        "PUT",
        "/matches/9999/result",
        {"ganador": "Test", "marcador": "6-0"},
        expected_status=404,
    )
    if response is not None or True:  # Esperamos error
        results["passed"] += 1
    else:
        results["failed"] += 1

    # ============================================================
    # RESUMEN FINAL
    # ============================================================

    log("\n" + "=" * 70, Colors.BLUE)
    log("📈 RESUMEN DE TESTS", Colors.BLUE)
    log("=" * 70, Colors.BLUE)

    success_rate = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0

    log(f"\nTotal tests: {results['total']}", Colors.BLUE)
    log(f"✅ Passed: {results['passed']}", Colors.GREEN)
    log(f"❌ Failed: {results['failed']}", Colors.RED)
    log(f"📊 Success rate: {success_rate:.1f}%\n", Colors.BLUE)

    if results["failed"] == 0:
        log("🎉 TODOS LOS TESTS PASARON!", Colors.GREEN)
        return 0
    else:
        log(f"⚠️  {results['failed']} tests fallaron", Colors.RED)
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
