#!/usr/bin/env python3
"""
Test script to verify AI components are working
Run: python test_ai.py
"""

import sys
import os
import json

def test_ai_components():
    print("="*60)
    print("🧪 AI COMPONENTS TEST")
    print("="*60)
    
    results = {}
    
    # Test 1: Sentence Transformers
    print("\n1️⃣  Testing Sentence Transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode("test string")
        print(f"   ✅ SUCCESS - Embedding shape: {test_embedding.shape}")
        results['sentence_transformers'] = True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results['sentence_transformers'] = False
    
    # Test 2: FAISS
    print("\n2️⃣  Testing FAISS...")
    try:
        import faiss
        import numpy as np
        dimension = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatL2(dimension)
        test_vector = np.random.random((1, dimension)).astype('float32')
        index.add(test_vector)
        print(f"   ✅ SUCCESS - Index size: {index.ntotal}")
        results['faiss'] = True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results['faiss'] = False
    
    # Test 3: Anthropic/Claude
    print("\n3️⃣  Testing Anthropic/Claude AI...")
    api_key = os.getenv('ANTHROPIC_API_KEY', '')
    if api_key and not api_key.startswith('sk-ant-your'):
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=50,
                messages=[{"role": "user", "content": "Reply with 'AI active' only"}]
            )
            result = response.content[0].text
            print(f"   ✅ SUCCESS - Claude response: {result}")
            results['anthropic'] = True
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results['anthropic'] = False
    else:
        print(f"   ⚠️  SKIPPED - No valid API key")
        results['anthropic'] = None
    
    # Test 4: AI Processor Module
    print("\n4️⃣  Testing AI Processor Module...")
    try:
        from dedupe_logic.ai_processor import AIDedupeProcessor
        processor = AIDedupeProcessor()
        status = processor.get_status()
        print(f"   ✅ Module loaded - Status: {status}")
        results['ai_processor'] = True
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        results['ai_processor'] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY:")
    print("="*60)
    
    for component, status in results.items():
        if status is True:
            print(f"✅ {component}: WORKING")
        elif status is False:
            print(f"❌ {component}: FAILED")
        else:
            print(f"⚠️  {component}: SKIPPED")
    
    # Overall status
    working = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    total = sum(1 for v in results.values() if v is not None)
    
    print("\n" + "-"*60)
    if failed == 0:
        print(f"🎉 AI LAYER OPERATIONAL ({working}/{total} components)")
        return 0
    else:
        print(f"❌ AI LAYER ISSUES ({failed} components failed)")
        print("\n📝 TO FIX:")
        if not results.get('sentence_transformers'):
            print("  - Install sentence-transformers: pip install sentence-transformers")
        if not results.get('faiss'):
            print("  - Install FAISS: pip install faiss-cpu")
        if results.get('anthropic') is None:
            print("  - Add Anthropic API key to .env file")
        return 1

if __name__ == "__main__":
    sys.exit(test_ai_components())
