#!/bin/bash

# Decode base64 credentials and create service account file
if [ -n "$GOOGLE_CREDENTIALS_BASE64" ]; then
    echo "🔑 Creating service account file from base64 credentials..."
    echo "$GOOGLE_CREDENTIALS_BASE64" | base64 -d > /app/service_account.json
    echo "✅ Service account file created"
else
    echo "⚠️ GOOGLE_CREDENTIALS_BASE64 not set"
fi

# Run the main application
echo "🚀 Starting Fuzzy Matching Pipeline..."
python main.py
