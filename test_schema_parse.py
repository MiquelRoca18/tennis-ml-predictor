#!/usr/bin/env python3
"""Test schema parsing to see what's happening"""

import re
from pathlib import Path

# Read schema
schema_path = Path("src/database/schema_v2.sql")
with open(schema_path, "r") as f:
    schema_script = f.read()

print("=== ORIGINAL SCHEMA LENGTH ===")
print(f"{len(schema_script)} characters")
print(f"First 500 chars:\n{schema_script[:500]}\n")

# Apply transformations
pg_schema = schema_script

# Replace AUTOINCREMENT
pg_schema = pg_schema.replace("AUTOINCREMENT", "")
pg_schema = pg_schema.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
pg_schema = pg_schema.replace("BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE")
pg_schema = pg_schema.replace("BOOLEAN DEFAULT 1", "BOOLEAN DEFAULT TRUE")

# Remove triggers
print("\n=== BEFORE TRIGGER REMOVAL ===")
print(f"Contains 'CREATE TRIGGER': {('CREATE TRIGGER' in pg_schema)}")

pg_schema = re.sub(r'CREATE TRIGGER.*?END;', '', pg_schema, flags=re.DOTALL)

print("\n=== AFTER TRIGGER REMOVAL ===")
print(f"Contains 'CREATE TRIGGER': {('CREATE TRIGGER' in pg_schema)}")
print(f"Length: {len(pg_schema)} characters")

# Remove test data
pg_schema = re.sub(r'-- Insertar partido de ejemplo.*$', '', pg_schema, flags=re.DOTALL)

print("\n=== AFTER TEST DATA REMOVAL ===")
print(f"Length: {len(pg_schema)} characters")

# Split statements
statements = [s.strip() for s in pg_schema.split(';') if s.strip() and not s.strip().startswith('--')]

print(f"\n=== TOTAL STATEMENTS: {len(statements)} ===")

# Show first 10 statements
for i, stmt in enumerate(statements[:10]):
    stmt_type = stmt.split()[0:3] if len(stmt.split()) >= 3 else stmt.split()
    print(f"{i+1}. {' '.join(stmt_type)} ... ({len(stmt)} chars)")

# Count CREATE TABLE statements
create_table_count = sum(1 for s in statements if s.upper().startswith('CREATE TABLE'))
print(f"\n=== CREATE TABLE COUNT: {create_table_count} ===")

# Show all CREATE TABLE statements
print("\n=== CREATE TABLE STATEMENTS ===")
for i, stmt in enumerate(statements):
    if stmt.upper().startswith('CREATE TABLE'):
        table_name = stmt.split()[5] if len(stmt.split()) > 5 else "unknown"
        print(f"Statement #{i+1}: {table_name}")
