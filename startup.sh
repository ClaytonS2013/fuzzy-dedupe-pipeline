#!/bin/bash

echo "🚀 Starting Fuzzy Matching Pipeline..."
echo "=================================="

# Run the main pipeline
python main.py

# Keep container alive if needed (for debugging)
# tail -f /dev/null
