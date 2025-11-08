#!/bin/bash

echo "🚀 Starting Fuzzy Matching Pipeline..."
echo "=================================="

# Debug: Check if environment variables are accessible
echo ""
echo "🔍 Checking environment variables..."
echo "GOOGLE_CREDENTIALS length: ${#GOOGLE_CREDENTIALS}"
echo "SUPABASE_URL: ${SUPABASE_URL:0:30}..."
echo "SPREADSHEET_ID: ${SPREADSHEET_ID}"

# Debug: List all env vars that contain our keywords
echo ""
echo "🔍 All relevant environment variables:"
env | grep -E "GOOGLE|SUPABASE|SPREADSHEET" | head -10

# Run the main pipeline
echo ""
echo "📊 Starting main pipeline..."
python main.py

# Keep container alive if needed (for debugging)
# tail -f /dev/null
