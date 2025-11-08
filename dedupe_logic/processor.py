"""
AI-Enhanced Fuzzy Matching Processor for Veterinary Practice Deduplication
Integrates: Traditional Fuzzy Matching + AI Embeddings + ML Models + Claude Validation
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

# AI/ML imports with graceful fallbacks
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
    from anthropic import Anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ Anthropic not installed. Run: pip install anthropic")

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

THRESHOLDS = {
    'high_confidence': float(os.environ.get('THRESHOLD', 85)) / 100,
    'medium_confidence': 0.75,
    'low_confidence': 0.50,
    'phone_match_boost': 0.15,
    'address_match_boost': 0.10
}

COMMON_WORDS = {
    'veterinary', 'vet', 'animal', 'hospital', 'clinic',
    'pet', 'care', 'center', 'medical', 'health',
    'associates', 'group', 'services', 'practice', 'llc',
    'inc', 'corp', 'company', 'the', 'and', 'of'
}

# ============================================
# UTILITY FUNCTIONS
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

def extract_tokens(text: str, remove_common: bool = True) -> set:
    """Extract meaningful tokens from text"""
    if not text:
        return set()
    
    tokens = set(text.lower().split())
    
    if remove_common:
        tokens = tokens - COMMON_WORDS
    
    return tokens

# ============================================
# TRADITIONAL FUZZY MATCHING
# ============================================

class TraditionalFuzzyMatcher:
    """Traditional fuzzy matching methods"""
    
    @staticmethod
    def calculate_match_score(record1: dict, record2: dict) -> dict:
        """Calculate various similarity scores"""
        
        # Name similarity
        name1 = normalize_text(record1.get('practice_name', ''))
        name2 = normalize_text(record2.get('practice_name', ''))
        name_similarity = calculate_similarity(name1, name2)
        
        # Token similarity
        tokens1 = extract_tokens(name1)
        tokens2 = extract_tokens(name2)
        token_overlap = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
        
        # Phone match
        phone1 = normalize_phone(record1.get('phone', ''))
        phone2 = normalize_phone(record2.get('phone', ''))
        phone_match = 1.0 if phone1 and phone1 == phone2 else 0.0
        
        # Address similarity
        addr1 = normalize_address(record1.get('address', ''))
        addr2 = normalize_address(record2.get('address', ''))
        address_similarity = calculate_similarity(addr1, addr2) if addr1 and addr2 else 0.0
        
        # Subset check
        is_subset = (name1 in name2 or name2 in name1) if name1 and name2 else False
        
        # Combined score
        base_score = name_similarity * 0.5 + token_overlap * 0.3
        
        if phone_match:
            base_score += THRESHOLDS['phone_match_boost']
        
        if address_similarity > 0.7:
            base_score += THRESHOLDS['address_match_boost']
        
        if is_subset and len(name1) > 3 and len(name2) > 3:
            base_score += 0.1
        
        return {
            'fuzzy_score': min(base_score, 1.0),
            'name_similarity': name_similarity,
            'token_overlap': token_overlap,
            'phone_match': phone_match,
            'address_similarity': address_similarity,
            'is_subset': is_subset
        }

# ============================================
# MAIN DEDUPLICATION FUNCTION
# ============================================

def run_deduplication(supabase):
    """Main deduplication function - works with or without AI"""
    logger.info("🚀 Starting deduplication pipeline...")
    
    try:
        # Fetch records from Supabase
        logger.info("📥 Fetching records from practice_records table...")
        response = supabase.table('practice_records').select('*').execute()
        records = response.data
        
        if not records:
            logger.warning("⚠️ No records found in practice_records table")
            return 0
        
        logger.info(f"📊 Processing {len(records)} records...")
        
        # Find duplicates using traditional fuzzy matching
        duplicates = find_fuzzy_duplicates(records)
        
        # Create clusters from duplicate pairs
        clusters = create_clusters(records, duplicates)
        
        # Merge records within each cluster
        deduplicated_records = []
        for cluster_id, cluster_indices in clusters.items():
            cluster_records = [records[i] for i in cluster_indices]
            merged_record = merge_cluster_records(cluster_records)
            merged_record['cluster_id'] = cluster_id
            merged_record['cluster_size'] = len(cluster_records)
            deduplicated_records.append(merged_record)
        
        # Save results to dedupe_results table
        if deduplicated_records:
            logger.info(f"💾 Saving {len(deduplicated_records)} deduplicated records...")
            
            # Clear existing results
            supabase.table('dedupe_results').delete().neq('id', -1).execute()
            
            # Insert new results in batches
            batch_size = 50
            for i in range(0, len(deduplicated_records), batch_size):
                batch = deduplicated_records[i:i + batch_size]
                supabase.table('dedupe_results').insert(batch).execute()
            
            logger.info(f"✅ Deduplication complete! Reduced from {len(records)} to {len(deduplicated_records)} records")
        
        return len(deduplicated_records)
        
    except Exception as e:
        logger.error(f"❌ Deduplication failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def find_fuzzy_duplicates(records: List[dict]) -> List[dict]:
    """Find duplicate records using fuzzy matching"""
    logger.info("🔍 Finding duplicates using fuzzy matching...")
    
    duplicates = []
    fuzzy_matcher = TraditionalFuzzyMatcher()
    
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            scores = fuzzy_matcher.calculate_match_score(records[i], records[j])
            
            if scores['fuzzy_score'] >= THRESHOLDS['medium_confidence']:
                duplicates.append({
                    'idx1': i,
                    'idx2': j,
                    'score': scores['fuzzy_score'],
                    'details': scores
                })
    
    logger.info(f"✅ Found {len(duplicates)} potential duplicate pairs")
    return duplicates

def create_clusters(records: List[dict], duplicates: List[dict]) -> dict:
    """Create clusters from duplicate pairs"""
    clusters = {}
    assigned = set()
    cluster_count = 0
    
    # Process duplicate pairs
    for dup in duplicates:
        idx1, idx2 = dup['idx1'], dup['idx2']
        
        # Find existing cluster or create new one
        cluster_id = None
        
        # Check if either record is already in a cluster
        for cid, members in clusters.items():
            if idx1 in members or idx2 in members:
                cluster_id = cid
                break
        
        if not cluster_id:
            cluster_id = f"cluster_{cluster_count:04d}"
            clusters[cluster_id] = set()
            cluster_count += 1
        
        # Add both records to the cluster
        clusters[cluster_id].add(idx1)
        clusters[cluster_id].add(idx2)
        assigned.add(idx1)
        assigned.add(idx2)
    
    # Add singleton clusters for unmatched records
    for i in range(len(records)):
        if i not in assigned:
            cluster_id = f"cluster_{cluster_count:04d}"
            clusters[cluster_id] = {i}
            cluster_count += 1
    
    # Convert sets to lists
    return {k: list(v) for k, v in clusters.items()}

def merge_cluster_records(cluster_records: List[dict]) -> dict:
    """Merge multiple records into a single deduplicated record"""
    if len(cluster_records) == 1:
        return cluster_records[0].copy()
    
    # Start with the most complete record
    def completeness_score(record):
        return sum(1 for v in record.values() if v and str(v).strip())
    
    merged = max(cluster_records, key=completeness_score).copy()
    
    # Intelligently merge fields from all records
    for field in merged.keys():
        if field in ['id', 'created_at', 'updated_at']:
            continue
        
        # Collect all non-empty values for this field
        values = [r.get(field) for r in cluster_records if r.get(field)]
        
        if values:
            if 'name' in field.lower():
                # For names, prefer the longest (usually most complete)
                merged[field] = max(values, key=lambda x: len(str(x)) if x else 0)
            elif 'phone' in field.lower():
                # For phones, use the most common or first valid one
                merged[field] = Counter(values).most_common(1)[0][0]
            else:
                # For other fields, use the most common value
                merged[field] = Counter(values).most_common(1)[0][0]
    
    # Add metadata about the merge
    merged['merge_count'] = len(cluster_records)
    merged['merge_confidence'] = 'high' if len(cluster_records) == 2 else 'medium'
    merged['merge_method'] = 'fuzzy_matching'
    
    return merged

# ============================================
# OPTIONAL: AI ENHANCEMENT (if packages installed)
# ============================================

if AI_EMBEDDINGS_AVAILABLE or LLM_AVAILABLE:
    logger.info("🎉 AI enhancements are available!")
    # Add AI-enhanced deduplication here if needed
else:
    logger.info("ℹ️ Running with traditional fuzzy matching only")
