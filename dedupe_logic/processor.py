# Add this to the imports section
import os
import sys

# Add this function to properly check AI
def check_ai_components():
    """Check and report AI component status"""
    ai_status = {
        'sentence_transformers': False,
        'faiss': False,
        'anthropic': False
    }
    
    try:
        import sentence_transformers
        ai_status['sentence_transformers'] = True
        print("✅ Sentence transformers loaded")
    except ImportError:
        print("❌ Sentence transformers NOT available - run: pip install sentence-transformers")
    
    try:
        import faiss
        ai_status['faiss'] = True
        print("✅ FAISS loaded")
    except ImportError:
        print("❌ FAISS NOT available - run: pip install faiss-cpu")
    
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if api_key and not api_key.startswith('sk-ant-your'):
        ai_status['anthropic'] = True
        print("✅ Anthropic API key configured")
    else:
        print("⚠️ Anthropic API key not configured")
    
    return ai_status

# Call this at module load
AI_STATUS = check_ai_components()
