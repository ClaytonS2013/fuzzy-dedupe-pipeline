#!/bin/bash

# AI Layer Fix Script
set -e

echo "🔧 AI LAYER FIX SCRIPT"
echo "======================"
echo ""

# Step 1: Check current status
echo "📊 Current Status:"
docker exec fuzzy-dedupe-pipeline python -c "
try:
    import sentence_transformers
    print('  ✅ Sentence Transformers: Installed')
except:
    print('  ❌ Sentence Transformers: Missing')

try:
    import faiss
    print('  ✅ FAISS: Installed')
except:
    print('  ❌ FAISS: Missing')

import os
api_key = os.getenv('ANTHROPIC_API_KEY', '')
if api_key and not api_key.startswith('sk-ant-your'):
    print('  ✅ Anthropic API: Configured')
else:
    print('  ⚠️  Anthropic API: Not configured')
" 2>/dev/null || echo "  ❌ Container not running or packages missing"

echo ""
echo "🔨 Starting fix process..."
echo ""

# Step 2: Rebuild with AI packages
echo "1️⃣  Rebuilding Docker image with AI packages..."
docker build --no-cache -t fuzzy-dedupe:ai-fixed . || {
    echo "❌ Build failed! Check Dockerfile"
    exit 1
}

echo ""
echo "2️⃣  Testing AI components in new image..."
docker run --rm -v $(pwd):/app fuzzy-dedupe:ai-fixed python test_ai.py || {
    echo "⚠️  Some AI components failed (this is normal if API key not set)"
}

echo ""
echo "3️⃣  Stopping old container..."
docker stop fuzzy-dedupe-pipeline 2>/dev/null || echo "  No container to stop"
docker rm fuzzy-dedupe-pipeline 2>/dev/null || echo "  No container to remove"

echo ""
echo "4️⃣  Starting new container with AI support..."
docker run -d \
    --name fuzzy-dedupe-pipeline \
    --env-file .env \
    --restart unless-stopped \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/models:/app/models \
    fuzzy-dedupe:ai-fixed

echo ""
echo "5️⃣  Waiting for container to start..."
sleep 5

echo ""
echo "6️⃣  Running AI component test..."
docker exec fuzzy-dedupe-pipeline python test_ai.py

echo ""
echo "✅ AI fix applied!"
echo ""
echo "📝 REMAINING STEPS:"
echo "1. Add your Anthropic API key to .env:"
echo "   ANTHROPIC_API_KEY=sk-ant-api-YOUR-KEY-HERE"
echo ""
echo "2. Fix database schema in Supabase SQL editor:"
echo "   ALTER TABLE practice_records"
echo "   ADD COLUMN IF NOT EXISTS canonical TEXT,"
echo "   ADD COLUMN IF NOT EXISTS reasoning TEXT;"
echo ""
echo "3. Check logs:"
echo "   docker logs -f fuzzy-dedupe-pipeline"
