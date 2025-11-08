#!/usr/bin/env python3
"""
Quick debug script to test environment variables
"""
import os
import sys

print("=" * 60)
print("🔍 ENVIRONMENT VARIABLE DEBUG")
print("=" * 60)

# Check each required variable
env_vars = [
    'SUPABASE_URL',
    'SUPABASE_KEY', 
    'SPREADSHEET_ID',
    'GOOGLE_CREDENTIALS'
]

for var in env_vars:
    value = os.environ.get(var)
    if value:
        if var == 'GOOGLE_CREDENTIALS':
            print(f"✅ {var}: Set (length: {len(value)} chars)")
            print(f"   First 20 chars: {value[:20]}...")
            print(f"   Last 20 chars: ...{value[-20:]}")
        else:
            print(f"✅ {var}: Set (length: {len(value)} chars)")
    else:
        print(f"❌ {var}: NOT SET or EMPTY")

print("=" * 60)
print("All environment variables:")
print("=" * 60)
for key in sorted(os.environ.keys()):
    if any(x in key.upper() for x in ['GOOGLE', 'SUPABASE', 'SPREADSHEET']):
        val = os.environ[key]
        if len(val) > 50:
            print(f"{key}: {val[:20]}...{val[-20:]}")
        else:
            print(f"{key}: {val}")

sys.exit(0)
