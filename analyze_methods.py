#!/usr/bin/env python3
"""
Script to automatically convert remaining SQLite methods to PostgreSQL-compatible versions
"""

import re

# Read the file
with open("src/database/match_database.py", "r") as f:
    content = f.read()

# Pattern to find methods that use self.conn.cursor() but aren't in the abstraction layer
# We'll convert them to use the abstraction methods

# List of methods to convert (excluding the ones already done and the abstraction layer itself)
methods_to_convert = [
    "update_match_live_data",
    "delete_match",
    "cleanup_old_matches",
    "add_prediction",
    "register_bet",
    "update_bet_result",
    "get_stats_summary",
    "get_daily_stats",
    "get_bet_history",
    "get_all_predictions_with_results",
]

# For now, let's just identify which methods still need conversion
print("Methods still using self.conn.cursor():")
for match in re.finditer(r'def (\w+)\(', content):
    method_name = match.group(1)
    # Find the method body
    start = match.start()
    # Find next method or end of class
    next_method = re.search(r'\n    def \w+\(', content[start+10:])
    if next_method:
        end = start + 10 + next_method.start()
    else:
        end = len(content)
    
    method_body = content[start:end]
    
    if 'self.conn.cursor()' in method_body and method_name not in ['_execute', '_fetchone', '_fetchall', '_init_sqlite']:
        print(f"  - {method_name}")

print("\nTotal methods to convert: Check output above")
