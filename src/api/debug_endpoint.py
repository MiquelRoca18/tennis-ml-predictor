
@app.get("/admin/debug-postgres", tags=["Admin"])
async def debug_postgres():
    """
    Endpoint de debug para probar PostgreSQL directamente
    
    Prueba:
    1. match_exists() con un partido que NO existe
    2. create_match() para insertar un partido de prueba
    3. match_exists() de nuevo (debería ser True)
    4. get_matches_by_date() para recuperar el partido
    5. Limpieza: eliminar el partido de prueba
    
    Returns:
        Resultados detallados de cada prueba
    """
    from datetime import date
    
    results = {
        "database_type": "PostgreSQL" if db.is_postgres else "SQLite",
        "tests": []
    }
    
    test_date = date.today()
    test_player1 = "DEBUG_TEST_PLAYER_1"
    test_player2 = "DEBUG_TEST_PLAYER_2"
    
    try:
        # TEST 1: Verificar que NO existe
        logger.info("🧪 TEST 1: Verificando match_exists() con partido inexistente...")
        exists_before = db.match_exists(test_player1, test_player2, test_date)
        results["tests"].append({
            "test": "1_match_exists_before",
            "expected": False,
            "actual": exists_before,
            "passed": not exists_before,
            "message": f"Partido debería NO existir. Result: {exists_before}"
        })
        
        # TEST 2: Crear partido
        logger.info("🧪 TEST 2: Creando partido de prueba...")
        try:
            match_id = db.create_match(
                fecha_partido=test_date,
                superficie="Hard",
                jugador1_nombre=test_player1,
                jugador1_cuota=1.5,
                jugador2_nombre=test_player2,
                jugador2_cuota=2.5,
                torneo="DEBUG_TEST_TOURNAMENT",
                hora_inicio="10:00",
                estado="pendiente"
            )
            results["tests"].append({
                "test": "2_create_match",
                "expected": "match_id > 0",
                "actual": match_id,
                "passed": match_id > 0,
                "message": f"Partido creado con ID: {match_id}"
            })
        except Exception as e:
            results["tests"].append({
                "test": "2_create_match",
                "expected": "success",
                "actual": str(e),
                "passed": False,
                "message": f"ERROR creando partido: {e}"
            })
            match_id = None
        
        # TEST 3: Verificar que AHORA existe
        logger.info("🧪 TEST 3: Verificando match_exists() después de crear...")
        exists_after = db.match_exists(test_player1, test_player2, test_date)
        results["tests"].append({
            "test": "3_match_exists_after",
            "expected": True,
            "actual": exists_after,
            "passed": exists_after,
            "message": f"Partido debería existir ahora. Result: {exists_after}"
        })
        
        # TEST 4: Recuperar con get_matches_by_date
        logger.info("🧪 TEST 4: Recuperando con get_matches_by_date()...")
        try:
            matches = db.get_matches_by_date(test_date)
            debug_matches = [m for m in matches if m.get('jugador1_nombre') == test_player1]
            results["tests"].append({
                "test": "4_get_matches_by_date",
                "expected": "1 match found",
                "actual": f"{len(debug_matches)} matches found (total: {len(matches)})",
                "passed": len(debug_matches) > 0,
                "message": f"Encontrados {len(debug_matches)} partidos de debug de {len(matches)} totales"
            })
        except Exception as e:
            results["tests"].append({
                "test": "4_get_matches_by_date",
                "expected": "success",
                "actual": str(e),
                "passed": False,
                "message": f"ERROR recuperando partidos: {e}"
            })
        
        # TEST 5: Contar partidos directamente con SQL
        logger.info("🧪 TEST 5: Contando partidos directamente con SQL...")
        try:
            count_result = db._fetchone(
                "SELECT COUNT(*) as count FROM matches WHERE jugador1_nombre = :player",
                {"player": test_player1}
            )
            count = count_result["count"] if count_result else 0
            results["tests"].append({
                "test": "5_direct_sql_count",
                "expected": "1",
                "actual": count,
                "passed": count > 0,
                "message": f"SQL directo encontró {count} partidos"
            })
        except Exception as e:
            results["tests"].append({
                "test": "5_direct_sql_count",
                "expected": "success",
                "actual": str(e),
                "passed": False,
                "message": f"ERROR en SQL directo: {e}"
            })
        
        # CLEANUP: Eliminar partido de prueba
        if match_id:
            logger.info("🧹 CLEANUP: Eliminando partido de prueba...")
            try:
                db.delete_match(match_id)
                results["cleanup"] = "✅ Partido de prueba eliminado"
            except Exception as e:
                results["cleanup"] = f"⚠️ Error eliminando: {e}"
        
        # Resumen
        passed = sum(1 for t in results["tests"] if t["passed"])
        total = len(results["tests"])
        results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": f"{passed}/{total} ({passed*100//total}%)"
        }
        
        logger.info(f"🧪 DEBUG TESTS COMPLETED: {passed}/{total} passed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error en debug endpoint: {e}", exc_info=True)
        return {
            "error": str(e),
            "database_type": "PostgreSQL" if db.is_postgres else "SQLite",
            "tests": results.get("tests", [])
        }
