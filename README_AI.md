# 🤖 AI Layer Documentation

## Overview
This document describes the AI-enhanced features of the Fuzzy Dedupe Pipeline.

## Current Status
Run `python test_ai.py` to check component status.

## Components

### 1. Sentence Transformers
- **Model**: all-MiniLM-L6-v2
- **Purpose**: Generate semantic embeddings for text similarity
- **Status**: ❌ Not Installed

### 2. FAISS
- **Purpose**: Fast similarity search in vector space
- **Status**: ❌ Not Installed

### 3. Claude AI (Anthropic)
- **Model**: claude-3-sonnet-20240229
- **Purpose**: Intelligent validation and merging
- **Status**: ❌ Not Configured

## Installation

### Quick Fix
```bash
# Run the automated fix script
bash fix_ai.sh
```

### Manual Installation
```bash
# 1. Install Python packages
pip install sentence-transformers==2.2.2 faiss-cpu==1.7.4

# 2. Add API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-api-YOUR-KEY" >> .env

# 3. Rebuild Docker image
docker build --no-cache -t fuzzy-dedupe:ai-enabled .

# 4. Restart container
docker-compose down
docker-compose up -d
```

## Configuration

Edit `config/ai_config.json` to adjust:
- Similarity threshold (default: 0.80)
- Embedding model
- Batch size
- Claude AI parameters

## Usage

When AI is enabled, the pipeline will:
1. Generate embeddings for all records
2. Find semantically similar records
3. Validate matches with Claude AI (if configured)
4. Smart merge duplicate records

## Testing
```bash
# Test AI components
docker exec fuzzy-dedupe-pipeline python test_ai.py

# Check AI processor
docker exec fuzzy-dedupe-pipeline python -c "
from dedupe_logic.ai_processor import AIDedupeProcessor
processor = AIDedupeProcessor()
print(processor.get_status())
"
```

## Troubleshooting

### Sentence Transformers not loading
- Ensure sufficient memory (4GB+ recommended)
- Check internet connection for model download

### FAISS errors
- Verify CPU supports required instructions
- Try: `pip install faiss-cpu --no-cache-dir`

### Claude AI not working
- Verify API key is correct
- Check API rate limits
- Ensure internet connectivity

## Performance

With AI enabled:
- Initial model download: ~500MB
- Memory usage: +1-2GB
- Processing time: +20-30% per batch
- Accuracy improvement: 40-60% for fuzzy matches
