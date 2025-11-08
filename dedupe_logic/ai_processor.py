"""
AI Enhancement Module for Fuzzy Dedupe Pipeline
Provides semantic matching and Claude AI validation
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class AIDedupeProcessor:
    def __init__(self):
        """Initialize AI components"""
        self.components = self._initialize_components()
        self.config = self._load_config()
        
    def _load_config(self):
        """Load AI configuration"""
        config_path = '/app/config/ai_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'ai_features': {
                'similarity_threshold': 0.80
            }
        }
    
    def _initialize_components(self):
        """Initialize available AI components"""
        components = {}
        
        # Try to load sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            model_name = 'all-MiniLM-L6-v2'
            cache_dir = '/app/models'
            os.makedirs(cache_dir, exist_ok=True)
            components['embedder'] = SentenceTransformer(model_name, cache_folder=cache_dir)
            logger.info(f"✅ Sentence transformer model loaded: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformers: {e}")
            components['embedder'] = None
        
        # Try to load FAISS
        try:
            import faiss
            components['faiss'] = faiss
            logger.info("✅ FAISS loaded for vector search")
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS: {e}")
            components['faiss'] = None
        
        # Try to load Anthropic
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key and not api_key.startswith('sk-ant-your'):
            try:
                from anthropic import Anthropic
                components['claude'] = Anthropic(api_key=api_key)
                logger.info("✅ Claude AI initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Claude: {e}")
                components['claude'] = None
        else:
            components['claude'] = None
            logger.info("ℹ️ Claude AI not configured (no valid API key)")
        
        return components
    
    def is_available(self) -> bool:
        """Check if AI features are available"""
        return any(self.components.values())
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of AI components"""
        return {
            'embeddings': self.components.get('embedder') is not None,
            'faiss': self.components.get('faiss') is not None,
            'claude': self.components.get('claude') is not None
        }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not self.components.get('embedder'):
            return 0.0
        
        try:
            embeddings = self.components['embedder'].encode([text1, text2])
            # Cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return 0.0
    
    def find_semantic_duplicates(self, records: List[Dict]) -> List[Tuple[int, int, float]]:
        """Find semantically similar records using embeddings"""
        if not self.components.get('embedder') or not self.components.get('faiss'):
            logger.warning("AI components not available for semantic matching")
            return []
        
        logger.info(f"🔍 Finding semantic duplicates in {len(records)} records...")
        
        # Create text representations
        texts = []
        for record in records:
            text = f"{record.get('name', '')} {record.get('address', '')} {record.get('city', '')}"
            texts.append(text)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.components['embedder'].encode(texts, show_progress_bar=False)
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_np.shape[1]
        index = self.components['faiss'].IndexFlatL2(dimension)
        index.add(embeddings_np)
        
        # Find similar pairs
        threshold = self.config['ai_features']['similarity_threshold']
        duplicates = []
        k = min(5, len(records))  # Search top-5 similar
        
        for i, embedding in enumerate(embeddings_np):
            distances, indices = index.search(embedding.reshape(1, -1), k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx != i and idx > i:  # Avoid self and duplicates
                    # Convert L2 distance to similarity
                    similarity = 1 / (1 + dist)
                    if similarity > threshold:
                        duplicates.append((i, idx, similarity))
                        logger.info(f"  🎯 Semantic match found: {texts[i][:50]} ↔ {texts[idx][:50]} ({similarity:.2%})")
        
        logger.info(f"✅ Found {len(duplicates)} semantic duplicate pairs")
        return duplicates
    
    def validate_with_claude(self, record1: Dict, record2: Dict, similarity: float) -> Dict:
        """Use Claude AI to validate if records are truly duplicates"""
        if not self.components.get('claude'):
            return {
                'is_duplicate': similarity > 0.85,
                'confidence': similarity,
                'method': 'threshold'
            }
        
        try:
            prompt = f"""Analyze if these two records represent the same entity:

Record 1:
{json.dumps(record1, indent=2)}

Record 2:
{json.dumps(record2, indent=2)}

Similarity Score: {similarity:.2%}

Respond with JSON only (no other text):
{{"is_duplicate": true_or_false, "confidence": 0.0_to_1.0, "reason": "brief_explanation"}}"""

            response = self.components['claude'].messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            result = json.loads(result_text)
            result['method'] = 'claude_ai'
            logger.info(f"🤖 Claude validation: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Claude validation error: {e}")
            return {
                'is_duplicate': similarity > 0.85,
                'confidence': similarity,
                'method': 'threshold',
                'error': str(e)
            }
    
    def smart_merge(self, records: List[Dict]) -> Dict:
        """Intelligently merge duplicate records"""
        if not records:
            return {}
        
        if self.components.get('claude'):
            try:
                prompt = f"""Merge these duplicate records into one, keeping the most complete and accurate information:

Records to merge:
{json.dumps(records, indent=2)}

Return ONLY the merged record as valid JSON, no other text."""

                response = self.components['claude'].messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                merged_text = response.content[0].text.strip()
                merged = json.loads(merged_text)
                merged['merge_method'] = 'claude_ai'
                logger.info("🤖 AI-powered merge completed")
                return merged
                
            except Exception as e:
                logger.error(f"AI merge error: {e}")
        
        # Fallback to simple merge
        merged = {}
        for record in records:
            for key, value in record.items():
                if value and (key not in merged or not merged[key]):
                    merged[key] = value
        merged['merge_method'] = 'simple'
        return merged
```

---

### **📁 `models/.gitkeep`** (NEW)
```
# This directory stores AI model files
# Models are downloaded automatically on first use
# Add actual model files to .gitignore
