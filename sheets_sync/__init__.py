# CRITICAL FIXES TO RUN:
cd fuzzy-dedupe-pipeline-main

# Fix 1: Rename init.py files to __init__.py
mv sheets_sync/init.py sheets_sync/__init__.py
mv dedupe_logic/init.py dedupe_logic/__init__.py

# Fix 2: Also check if database needs __init__.py
touch database/__init__.py
