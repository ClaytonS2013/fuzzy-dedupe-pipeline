"""
AI-Enhanced Fuzzy Matching Processor for Veterinary Practice Deduplication
Integrates: Traditional Fuzzy Matching + AI Embeddings + ML Models + LLM Validation
"""

import os
import re
import json
import logging
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from datetime import datetime

# AI/ML imports (install these first)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    AI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    AI_EMBEDDINGS_AVAILABLE = False
    print("⚠️ Sentence transformers not installed. Run: pip install sentence-transformers faiss-cpu")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Scikit-learn not installed. Run: pip install scikit-learn")

try:
    import openai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ OpenAI not installed. Run: pip install openai")

logger = logging.getLogger(__name__)

# ============================================
# PART 1: TRADITIONAL FUZZY MATCHING (Existing)
# ============================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison"""
    if not phone:
        return ""
    return re.sub(r'\D', '', str(phone))

def normalize_address(address: str) -> str:
    """Normalize address for comparison"""
    if not address:
        return ""
    
    address = str(address).lower().strip()
    replacements = {
        'street': 'st', 'avenue': 'ave', 'road': 'rd',
        'boulevard': 'blvd', 'drive': 'dr', 'court': 'ct',
        'place': 'pl', 'lane': 'ln', 'suite': 'ste',
        'north': 'n', 'south': 's', 'east': 'e', 'west': 'w'
    }
    
    for full, abbr in replacements.items():
        address = address.replace(full, abbr)
    
    address = re.sub(r'[^\w\s]', ' ', address)
    address = re.sub(r'\s+', ' ', address)
    return address

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()

# ============================================
# PART 2: AI EMBEDDINGS MATCHING
# ============================================

class AIEmbeddingMatcher:
    """AI-powered semantic similarity matching using embeddings"""
    
    def __init__(self):
        if AI_EMBEDDINGS_AVAILABLE:
            logger.info("🧠 Loading AI embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ AI embedding model loaded")
        else:
            self.model = None
            logger.warning("⚠️ AI embeddings not available")
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for a single text"""
        if not self.model:
            return None
        return self.model.encode([text])[0]
    
    def find_semantic_duplicates(self, records: List[dict], threshold: float = 0.80) -> List[dict]:
        """Find duplicates using semantic similarity"""
        if not self.model:
            return []
        
        logger.info("🧠 Computing semantic embeddings...")
        
        # Create text representations for each record
        texts = []
        for r in records:
            text = f"{r.get('practice_name', '')} {r.get('address', '')} {r.get('city', '')}"
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index for fast similarity search
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Find similar pairs
        semantic_matches = []
        k = min(10, len(records))  # Find top k similar
        
        similarities, indices = index.search(embeddings, k)
        
        for i in range(len(records)):
            for j_idx, sim in zip(indices[i][1:], similarities[i][1:]):
                if sim > threshold and i < j_idx:  # Avoid duplicate pairs
                    semantic_matches.append({
                        'idx1': i,
                        'idx2': j_idx,
                        'record1': records[i],
                        'record2': records[j_idx],
                        'semantic_score': float(sim),
                        'match_type': 'semantic'
                    })
        
        logger.info(f"✅ Found {len(semantic_matches)} semantic matches")
        return semantic_matches

# ============================================
# PART 3: MACHINE LEARNING MODEL
# ============================================

class MLDuplicatePredictor:
    """Machine learning model for duplicate prediction"""
    
    def __init__(self, model_path: str = 'dedupe_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'name_similarity', 'token_similarity', 'phone_match',
            'address_similarity', 'name_length_diff', 'common_words'
        ]
        self.load_model()
    
    def load_model(self):
        """Load existing model if available"""
        if os.path.exists(self.model_path) and ML_AVAILABLE:
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ ML model loaded from {self.model_path}")
            except:
                self.model = None
        elif ML_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def extract_features(self, record1: dict, record2: dict) -> dict:
        """Extract ML features from record pair"""
        name1 = record1.get('practice_name', '')
        name2 = record2.get('practice_name', '')
        
        features = {
            'name_similarity': calculate_similarity(name1, name2),
            'token_similarity': self._token_similarity(name1, name2),
            'phone_match': 1.0 if normalize_phone(record1.get('phone')) == normalize_phone(record2.get('phone')) else 0.0,
            'address_similarity': calculate_similarity(
                normalize_address(record1.get('address', '')),
                normalize_address(record2.get('address', ''))
            ),
            'name_length_diff': min(abs(len(name1) - len(name2)) / 50.0, 1.0),
            'common_words': len(set(name1.lower().split()) & set(name2.lower().split())) / 10.0
        }
        return features
    
    def _token_similarity(self, s1: str, s2: str) -> float:
        """Calculate token-based similarity"""
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union) if union else 0.0
    
    def predict(self, record1: dict, record2: dict) -> Tuple[bool, float]:
        """Predict if two records are duplicates"""
        if not self.model or not ML_AVAILABLE:
            return None, 0.0
        
        features = self.extract_features(record1, record2)
        X = [[features[name] for name in self.feature_names]]
        
        try:
            probability = self.model.predict_proba(X)[0][1]
            is_duplicate = probability > 0.5
            return is_duplicate, probability
        except:
            # Model not trained yet
            return None, 0.0
    
    def train(self, training_data: List[Tuple[dict, dict, bool]]):
        """Train model on labeled data"""
        if not ML_AVAILABLE:
            return
        
        X = []
        y = []
        
        for record1, record2, is_duplicate in training_data:
            features = self.extract_features(record1, record2)
            X.append([features[name] for name in self.feature_names])
            y.append(1 if is_duplicate else 0)
        
        if len(X) > 10:  # Need minimum samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"✅ ML model trained with {accuracy:.2%} accuracy")
            return accuracy
        return 0.0

# ============================================
# PART 4: LLM VALIDATION
# ============================================

class LLMValidator:
    """Large Language Model validation for complex cases"""
    
    def __init__(self):
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if self.api_key and LLM_AVAILABLE:
            openai.api_key = self.api_key
            self.enabled = True
            logger.info("✅ LLM validation enabled")
        else:
            self.enabled = False
            logger.info("ℹ️ LLM validation not configured")
    
    def validate_match(self, record1: dict, record2: dict, current_score: float) -> dict:
        """Use LLM to validate a potential match"""
        if not self.enabled:
            return {'use_llm': False}
        
        # Only use LLM for uncertain cases (0.5 to 0.85 confidence)
        if current_score < 0.5 or current_score > 0.85:
            return {'use_llm': False}
        
        prompt = f"""
        Analyze if these veterinary practices are the same entity:
        
        Practice 1:
        Name: {record1.get('practice_name')}
        Phone: {record1.get('phone')}
        Address: {record1.get('address')}
        City: {record1.get('city')}
        
        Practice 2:
        Name: {record2.get('practice_name')}
        Phone: {record2.get('phone')}
        Address: {record2.get('address')}
        City: {record2.get('city')}
        
        Current match score: {current_score:.2%}
        
        Consider: name variations, common abbreviations, chain locations, acquisitions.
        
        Respond with JSON only:
        {{"is_duplicate": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use gpt-4 for better accuracy
                messages=[
                    {"role": "system", "content": "You are an expert at identifying duplicate veterinary practice records. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = json.loads(response.choices[0].message.content)
            result['use_llm'] = True
            return result
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return {'use_llm': False}

# ============================================
# PART 5: INTEGRATED AI DEDUPLICATION
# ============================================

class AIEnhancedDeduplicator:
    """Main class that integrates all deduplication methods"""
    
    def __init__(self):
        self.embedding_matcher = AIEmbeddingMatcher() if AI_EMBEDDINGS_AVAILABLE else None
        self.ml_predictor = MLDuplicatePredictor() if ML_AVAILABLE else None
        self.llm_validator = LLMValidator() if LLM_AVAILABLE else None
        
        # Weights for combining different signals
        self.weights = {
            'fuzzy': 0.30,
            'semantic': 0.30,
            'ml': 0.25,
            'llm': 0.15
        }
    
    def find_all_duplicates(self, records: List[dict]) -> List[dict]:
        """Find duplicates using all available methods"""
        all_matches = {}
        
        logger.info("🔍 Starting AI-enhanced deduplication...")
        
        # 1. Traditional fuzzy matching
        fuzzy_matches = self._fuzzy_match_all(records)
        for match in fuzzy_matches:
            key = (match['idx1'], match['idx2'])
            all_matches[key] = match
            match['scores'] = {'fuzzy': match['fuzzy_score']}
        
        # 2. AI Semantic matching
        if self.embedding_matcher:
            semantic_matches = self.embedding_matcher.find_semantic_duplicates(records)
            for match in semantic_matches:
                key = (match['idx1'], match['idx2'])
                if key in all_matches:
                    all_matches[key]['scores']['semantic'] = match['semantic_score']
                else:
                    all_matches[key] = match
                    match['scores'] = {'semantic': match['semantic_score']}
        
        # 3. ML predictions
        if self.ml_predictor:
            for key, match in all_matches.items():
                is_dup, prob = self.ml_predictor.predict(
                    match['record1'], match['record2']
                )
                if prob > 0:
                    match['scores']['ml'] = prob
        
        # 4. Calculate combined scores
        final_matches = []
        for match in all_matches.values():
            combined_score = self._calculate_combined_score(match['scores'])
            match['combined_score'] = combined_score
            
            # 5. LLM validation for uncertain cases
            if self.llm_validator and 0.5 < combined_score < 0.85:
                llm_result = self.llm_validator.validate_match(
                    match['record1'], match['record2'], combined_score
                )
                if llm_result.get('use_llm'):
                    match['scores']['llm'] = llm_result['confidence']
                    match['llm_reason'] = llm_result.get('reason', '')
                    match['combined_score'] = self._calculate_combined_score(match['scores'])
            
            # Final decision
            match['is_duplicate'] = match['combined_score'] > 0.75
            final_matches.append(match)
        
        # Sort by confidence
        final_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"✅ Found {sum(1 for m in final_matches if m['is_duplicate'])} duplicate pairs")
        return final_matches
    
    def _fuzzy_match_all(self, records: List[dict]) -> List[dict]:
        """Traditional fuzzy matching for all records"""
        matches = []
        
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                score = self._fuzzy_match_pair(records[i], records[j])
                if score > 0.5:  # Lower threshold to catch more candidates
                    matches.append({
                        'idx1': i,
                        'idx2': j,
                        'record1': records[i],
                        'record2': records[j],
                        'fuzzy_score': score,
                        'match_type': 'fuzzy'
                    })
        
        return matches
    
    def _fuzzy_match_pair(self, r1: dict, r2: dict) -> float:
        """Calculate fuzzy match score for a pair"""
        name_sim = calculate_similarity(
            normalize_text(r1.get('practice_name', '')),
            normalize_text(r2.get('practice_name', ''))
        )
        
        phone_match = 0.2 if normalize_phone(r1.get('phone')) == normalize_phone(r2.get('phone')) else 0.0
        
        addr_sim = calculate_similarity(
            normalize_address(r1.get('address', '')),
            normalize_address(r2.get('address', ''))
        ) * 0.3
        
        return min(name_sim + phone_match + addr_sim, 1.0)
    
    def _calculate_combined_score(self, scores: dict) -> float:
        """Calculate weighted combination of all scores"""
        total_score = 0
        total_weight = 0
        
        for method, score in scores.items():
            weight = self.weights.get(method, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

# ============================================
# MAIN ENTRY POINT
# ============================================

def run_deduplication(supabase):
    """Main deduplication function with AI enhancements"""
    logger.info("🚀 Starting AI-enhanced deduplication pipeline...")
    
    try:
        # Fetch records
        logger.info("📥 Fetching records from practice_records table...")
        response = supabase.table('practice_records').select('*').execute()
        records = response.data
        
        if not records:
            logger.warning("⚠️ No records found")
            return 0
        
        logger.info(f"📊 Processing {len(records)} records...")
        
        # Run AI-enhanced deduplication
        deduplicator = AIEnhancedDeduplicator()
        matches = deduplicator.find_all_duplicates(records)
        
        # Create clusters from matches
        clusters = _create_clusters_from_matches(records, matches)
        
        # Merge duplicates
        deduplicated_records = []
        for cluster_id, cluster_indices in clusters.items():
            cluster_records = [records[i] for i in cluster_indices]
            merged_record = _merge_cluster(cluster_records)
            merged_record['cluster_id'] = cluster_id
            merged_record['cluster_size'] = len(cluster_records)
            deduplicated_records.append(merged_record)
        
        # Save results
        if deduplicated_records:
            logger.info(f"💾 Saving {len(deduplicated_records)} deduplicated records...")
            
            # Clear existing results
            supabase.table('dedupe_results').delete().neq('id', -1).execute()
            
            # Insert new results
            batch_size = 50
            for i in range(0, len(deduplicated_records), batch_size):
                batch = deduplicated_records[i:i + batch_size]
                supabase.table('dedupe_results').insert(batch).execute()
            
            logger.info("✅ AI-enhanced deduplication complete!")
        
        # Log statistics
        stats = {
            'original_count': len(records),
            'deduplicated_count': len(deduplicated_records),
            'duplicates_removed': len(records) - len(deduplicated_records),
            'ai_features': {
                'embeddings': AI_EMBEDDINGS_AVAILABLE,
                'ml_model': ML_AVAILABLE and deduplicator.ml_predictor.model is not None,
                'llm_validation': LLM_AVAILABLE and deduplicator.llm_validator.enabled
            }
        }
        
        logger.info(f"📈 Statistics: {json.dumps(stats, indent=2)}")
        return len(deduplicated_records)
        
    except Exception as e:
        logger.error(f"❌ Deduplication failed: {e}")
        raise

def _create_clusters_from_matches(records: List[dict], matches: List[dict]) -> dict:
    """Create record clusters from pairwise matches"""
    clusters = {}
    assigned = set()
    cluster_count = 0
    
    # Process confirmed duplicates
    for match in matches:
        if not match['is_duplicate']:
            continue
            
        idx1, idx2 = match['idx1'], match['idx2']
        
        # Find or create cluster
        cluster_id = None
        if idx1 in assigned:
            for cid, members in clusters.items():
                if idx1 in members:
                    cluster_id = cid
                    break
        
        if not cluster_id:
            cluster_id = f"cluster_{cluster_count:04d}"
            clusters[cluster_id] = set()
            cluster_count += 1
        
        clusters[cluster_id].add(idx1)
        clusters[cluster_id].add(idx2)
        assigned.add(idx1)
        assigned.add(idx2)
    
    # Add singletons
    for i in range(len(records)):
        if i not in assigned:
            cluster_id = f"cluster_{cluster_count:04d}"
            clusters[cluster_id] = {i}
            cluster_count += 1
    
    # Convert sets to lists
    return {k: list(v) for k, v in clusters.items()}

def _merge_cluster(cluster_records: List[dict]) -> dict:
    """Merge multiple records into one"""
    if len(cluster_records) == 1:
        return cluster_records[0].copy()
    
    # Start with most complete record
    def completeness(record):
        return sum(1 for v in record.values() if v and str(v).strip())
    
    merged = max(cluster_records, key=completeness).copy()
    
    # Merge fields intelligently
    for key in merged.keys():
        if key in ['id', 'created_at', 'updated_at']:
            continue
        
        values = [r.get(key) for r in cluster_records if r.get(key)]
        if values:
            if 'name' in key.lower():
                # Choose longest name (usually most complete)
                merged[key] = max(values, key=lambda x: len(str(x)) if x else 0)
            else:
                # Choose most common value
                value_counts = Counter(values)
                merged[key] = value_counts.most_common(1)[0][0]
    
    merged['merge_count'] = len(cluster_records)
    merged['merge_confidence'] = 'ai_enhanced'
    
    return merged
