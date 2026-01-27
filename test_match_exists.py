# Test para verificar por qué match_exists devuelve True para todo
import requests
from datetime import date

# Probar con un partido que definitivamente NO existe
response = requests.get("https://tennis-ml-predictor-production.up.railway.app/admin/debug-postgres")
data = response.json()

print("=== ANÁLISIS DE match_exists() ===")
print(f"Database type: {data.get('database_type')}")
print(f"\nTest 1 (match_exists before): {data['tests'][0]}")
print(f"\nProblema: match_exists() devuelve {data['tests'][0]['actual']} cuando debería devolver False")
print(f"\nEsto significa que la query de match_exists() está:")
print("  A) Devolviendo resultados cuando no debería, O")
print("  B) Fallando silenciosamente y devolviendo True por defecto")
