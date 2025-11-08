#!/bin/bash

echo "🚀 Starting Fuzzy Matching Pipeline..."
echo "=================================="
echo "Time: $(date)"
echo "=================================="

# Ensure Python can find modules
export PYTHONPATH="${PYTHONPATH}:/app"

# Debug: Check if environment variables are accessible
echo ""
echo "🔍 Checking environment variables..."

if [ -z "$GOOGLE_CREDENTIALS" ]; then
    echo "❌ ERROR: GOOGLE_CREDENTIALS is not set!"
else
    echo "✅ GOOGLE_CREDENTIALS length: ${#GOOGLE_CREDENTIALS}"
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "❌ ERROR: SUPABASE_URL is not set!"
else
    echo "✅ SUPABASE_URL: ${SUPABASE_URL:0:30}..."
fi

if [ -z "$SUPABASE_KEY" ]; then
    echo "❌ ERROR: SUPABASE_KEY is not set!"
else
    echo "✅ SUPABASE_KEY is set (${#SUPABASE_KEY} chars)"
fi

if [ -z "$SPREADSHEET_ID" ]; then
    echo "❌ ERROR: SPREADSHEET_ID is not set!"
else
    echo "✅ SPREADSHEET_ID: ${SPREADSHEET_ID}"
fi

# Debug: List all env vars that contain our keywords
echo ""
echo "🔍 All relevant environment variables:"
env | grep -E "GOOGLE|SUPABASE|SPREADSHEET" | while read line; do
    var_name=$(echo "$line" | cut -d'=' -f1)
    var_value=$(echo "$line" | cut -d'=' -f2)
    if [ ${#var_value} -gt 50 ]; then
        echo "  $var_name=${var_value:0:20}...${var_value: -20}"
    else
        echo "  $var_name=$var_value"
    fi
done

# Pre-import required Python modules to warm up the environment
echo ""
echo "📦 Pre-loading Python modules..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
print('Loading modules...')
try:
    import json
    print('✅ json loaded')
except ImportError as e:
    print(f'❌ json failed: {e}')
    
try:
    import base64
    print('✅ base64 loaded')
except ImportError as e:
    print(f'❌ base64 failed: {e}')
    
try:
    import gspread
    print('✅ gspread loaded')
except ImportError as e:
    print(f'❌ gspread failed: {e}')
    
try:
    from supabase import create_client
    print('✅ supabase loaded')
except ImportError as e:
    print(f'❌ supabase failed: {e}')
    
print('Module loading complete!')
" || echo "⚠️ Warning: Module pre-loading failed"

# Run the main pipeline
echo ""
echo "📊 Starting main pipeline..."
echo "=================================="

# Run with explicit error handling
python3 main.py
PIPELINE_EXIT_CODE=$?

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Pipeline completed successfully!"
    echo "=================================="
else
    echo ""
    echo "❌ Pipeline failed with exit code: $PIPELINE_EXIT_CODE"
    echo "=================================="
    exit $PIPELINE_EXIT_CODE
fi

# Keep container alive if needed (for debugging)
# tail -f /dev/null
