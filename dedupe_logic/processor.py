"""
Comprehensive Fuzzy Matching Processor for Veterinary Practice Deduplication
Ready-to-use implementation with multiple matching strategies
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
import json

# Using built-in libraries to avoid additional dependencies
from difflib import SequenceMatcher
from collections import defaultdict

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    # Convert to lowercase, remove extra spaces, remove special chars
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison"""
    if not phone:
        return ""
    # Keep only digits
    return re.sub(r'\D', '', str(phone))


def normalize_address(address: str) -> str:
    """Normalize address for comparison"""
    if not address:
        return ""
    
    address = str(address).lower().strip()
    
    # Common address abbreviations
    replacements = {
        'street': 'st',
        'avenue': 'ave',
        'road': 'rd',
        'boulevard': 'blvd',
        'drive': 'dr',
        'court': 'ct',
        'place': 'pl',
        'lane': 'ln',
        'suite': 'ste',
        'north': 'n',
        'south': 's',
        'east': 'e',
        'west': 'w',
        'apartment': 'apt',
        'building': 'bldg'
    }
    
    for full, abbr in replacements.items():
        address = address.replace(full, abbr)
    
    # Remove punctuation and extra spaces
    address = re.sub(r'[^\w\s]', ' ', address)
    address = re.sub(r'\s+', ' ', address)
    
    return address


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher"""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()


def extract_name_tokens(name: str) -> set:
    """Extract important tokens from practice name"""
    name = normalize_text(name)
    
    # Common words to ignore in veterinary practice names
    stop_words = {
        'veterinary', 'vet', 'clinic', 'hospital', 'animal', 'pet', 
        'care', 'center', 'centre', 'practice', 'associates', 'group',
        'the', 'and', 'of', 'llc', 'inc', 'ltd', 'pa', 'pc'
    }
    
    tokens = set(name.split())
    return tokens - stop_words


def calculate_token_similarity(name1: str, name2: str) -> float:
    """Calculate similarity based on shared important tokens"""
    tokens1 = extract_name_tokens(name1)
    tokens2 = extract_name_tokens(name2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    return len(intersection) / len(union) if union else 0.0


def is_likely_duplicate(record1: dict, record2: dict, thresholds: dict = None) -> Tuple[bool, float, str]:
    """
    Determine if two records are likely duplicates
    Returns: (is_duplicate, confidence_score, match_reason)
    """
    
    if thresholds is None:
        thresholds = {
            'exact_match': 1.0,
            'high_confidence': 0.85,
            'medium_confidence': 0.75,
            'phone_match_boost': 0.15,
            'address_match_boost': 0.10
        }
    
    # Extract and normalize fields
    name1 = normalize_text(record1.get('practice_name', ''))
    name2 = normalize_text(record2.get('practice_name', ''))
    
    phone1 = normalize_phone(record1.get('phone', ''))
    phone2 = normalize_phone(record2.get('phone', ''))
    
    address1 = normalize_address(record1.get('address', ''))
    address2 = normalize_address(record2.get('address', ''))
    
    # Calculate base name similarity
    name_similarity = calculate_similarity(name1, name2)
    token_similarity = calculate_token_similarity(
        record1.get('practice_name', ''),
        record2.get('practice_name', '')
    )
    
    # Use the higher of the two name similarity scores
    best_name_score = max(name_similarity, token_similarity)
    
    # Start with name score
    total_score = best_name_score
    match_reasons = []
    
    # Exact name match
    if name1 and name1 == name2:
        match_reasons.append("exact_name")
        return (True, 1.0, "exact_name_match")
    
    # Strong name similarity
    if best_name_score >= thresholds['high_confidence']:
        match_reasons.append(f"high_name_similarity_{best_name_score:.2f}")
    
    # Phone number match (strong indicator)
    if phone1 and phone2 and len(phone1) >= 10 and phone1 == phone2:
        total_score += thresholds['phone_match_boost']
        match_reasons.append("phone_match")
        
        # Phone match with moderate name similarity is very likely a duplicate
        if best_name_score >= 0.6:
            return (True, min(total_score, 1.0), "phone_match_with_similar_name")
    
    # Address similarity
    if address1 and address2:
        address_similarity = calculate_similarity(address1, address2)
        if address_similarity >= 0.8:
            total_score += thresholds['address_match_boost']
            match_reasons.append(f"address_match_{address_similarity:.2f}")
            
            # Address match with moderate name similarity is likely a duplicate
            if best_name_score >= 0.6:
                return (True, min(total_score, 1.0), "address_match_with_similar_name")
    
    # Check for subset relationships (one name contains the other)
    if name1 and name2:
        if (name1 in name2 or name2 in name1) and len(name1) > 5 and len(name2) > 5:
            total_score = max(total_score, 0.8)
            match_reasons.append("name_subset")
    
    # Final decision based on total score
    is_duplicate = total_score >= thresholds['medium_confidence']
    match_reason = ", ".join(match_reasons) if match_reasons else "no_significant_match"
    
    return (is_duplicate, min(total_score, 1.0), match_reason)


def cluster_duplicates(records: List[dict]) -> Dict[str, List[int]]:
    """
    Cluster duplicate records together
    Returns a dictionary mapping cluster_id to list of record indices
    """
    n = len(records)
    clusters = {}
    assigned = set()
    cluster_count = 0
    
    logger.info(f"🔍 Comparing {n} records for duplicates...")
    
    # Compare all pairs of records
    matches_found = []
    for i in range(n):
        if i in assigned:
            continue
            
        cluster_id = f"cluster_{cluster_count + 1:03d}"
        cluster_members = [i]
        
        for j in range(i + 1, n):
            if j in assigned:
                continue
                
            is_dup, confidence, reason = is_likely_duplicate(records[i], records[j])
            
            if is_dup:
                cluster_members.append(j)
                assigned.add(j)
                matches_found.append({
                    'record1': records[i].get('practice_name', 'Unknown'),
                    'record2': records[j].get('practice_name', 'Unknown'),
                    'confidence': confidence,
                    'reason': reason
                })
                logger.info(f"  ✓ Match found: {records[i].get('practice_name', 'Unknown')[:30]} "
                          f"== {records[j].get('practice_name', 'Unknown')[:30]} "
                          f"(confidence: {confidence:.2f}, reason: {reason})")
        
        # Only create cluster if we found duplicates
        if len(cluster_members) > 1:
            clusters[cluster_id] = cluster_members
            cluster_count += 1
            assigned.update(cluster_members)
        
    logger.info(f"📊 Found {len(matches_found)} duplicate pairs forming {len(clusters)} clusters")
    
    # Assign unique clusters to remaining records
    for i in range(n):
        if i not in assigned:
            cluster_id = f"cluster_{cluster_count + 1:03d}"
            clusters[cluster_id] = [i]
            cluster_count += 1
    
    return clusters


def merge_duplicate_records(cluster_records: List[dict]) -> dict:
    """
    Merge multiple duplicate records into one canonical record
    Uses the most complete and most recent information
    """
    if not cluster_records:
        return {}
    
    if len(cluster_records) == 1:
        return cluster_records[0].copy()
    
    # Start with the record that has the most non-empty fields
    def count_fields(record):
        return sum(1 for v in record.values() if v and str(v).strip())
    
    canonical = max(cluster_records, key=count_fields).copy()
    
    # For each field, choose the best value from all records
    all_keys = set()
    for record in cluster_records:
        all_keys.update(record.keys())
    
    for key in all_keys:
        if key in ['id', 'created_at', 'updated_at']:  # Skip metadata fields
            continue
            
        values = [r.get(key) for r in cluster_records if r.get(key)]
        
        if values:
            # For name fields, choose the longest (usually most complete)
            if 'name' in key.lower():
                canonical[key] = max(values, key=lambda x: len(str(x)) if x else 0)
            # For other fields, choose the most common or the longest
            else:
                # Most common value
                from collections import Counter
                value_counts = Counter(values)
                canonical[key] = value_counts.most_common(1)[0][0]
    
    # Add metadata about the merge
    canonical['merge_count'] = len(cluster_records)
    canonical['merge_confidence'] = 'high' if len(cluster_records) == 2 else 'very_high'
    
    return canonical


def run_deduplication(supabase):
    """
    Main deduplication function that integrates with the pipeline
    """
    logger.info("🚀 Starting comprehensive fuzzy matching deduplication...")
    
    try:
        # Fetch all records from practice_records
        logger.info("📥 Fetching records from practice_records table...")
        response = supabase.table('practice_records').select('*').execute()
        records = response.data
        
        if not records:
            logger.warning("⚠️ No records found in practice_records table")
            return 0
        
        logger.info(f"📊 Processing {len(records)} records for deduplication...")
        
        # Step 1: Find and cluster duplicates
        clusters = cluster_duplicates(records)
        
        # Step 2: Merge duplicates within each cluster
        deduplicated_records = []
        duplicate_count = 0
        
        for cluster_id, indices in clusters.items():
            cluster_records = [records[i] for i in indices]
            
            if len(cluster_records) > 1:
                duplicate_count += len(cluster_records) - 1
                logger.info(f"  🔀 Merging {len(cluster_records)} records in {cluster_id}")
            
            # Merge the cluster into a single record
            merged_record = merge_duplicate_records(cluster_records)
            merged_record['cluster_id'] = cluster_id
            merged_record['cluster_size'] = len(cluster_records)
            
            deduplicated_records.append(merged_record)
        
        logger.info(f"✨ Deduplication complete: {len(records)} → {len(deduplicated_records)} records")
        logger.info(f"🎯 Removed {duplicate_count} duplicate records")
        
        # Step 3: Save results to dedupe_results table
        if deduplicated_records:
            logger.info(f"💾 Saving {len(deduplicated_records)} deduplicated records to dedupe_results...")
            
            # Clear existing results
            supabase.table('dedupe_results').delete().neq('id', -1).execute()
            
            # Insert new results in batches (Supabase has limits)
            batch_size = 50
            for i in range(0, len(deduplicated_records), batch_size):
                batch = deduplicated_records[i:i + batch_size]
                supabase.table('dedupe_results').insert(batch).execute()
            
            logger.info("✅ Deduplicated records saved to dedupe_results table")
        
        # Log statistics
        stats = {
            'original_count': len(records),
            'deduplicated_count': len(deduplicated_records),
            'duplicates_removed': duplicate_count,
            'clusters_formed': len([c for c in clusters.values() if len(c) > 1]),
            'reduction_percentage': round((duplicate_count / len(records)) * 100, 2) if records else 0
        }
        
        logger.info(f"📈 Deduplication Statistics: {json.dumps(stats, indent=2)}")
        
        return len(deduplicated_records)
        
    except Exception as e:
        logger.error(f"❌ Deduplication failed: {e}")
        raise


# For testing individual functions
if __name__ == "__main__":
    # Test with sample veterinary practice data
    test_records = [
        {'practice_name': 'ABC Veterinary Hospital', 'phone': '555-1234', 'address': '123 Main St'},
        {'practice_name': 'ABC Vet Hospital', 'phone': '5551234', 'address': '123 Main Street'},
        {'practice_name': 'XYZ Animal Clinic', 'phone': '555-5678', 'address': '456 Oak Ave'},
        {'practice_name': 'ABC Veterinary', 'phone': '555-1234', 'address': '123 Main'},
    ]
    
    print("Testing duplicate detection...")
    for i in range(len(test_records)):
        for j in range(i + 1, len(test_records)):
            is_dup, conf, reason = is_likely_duplicate(test_records[i], test_records[j])
            if is_dup:
                print(f"✓ Duplicates found: Record {i+1} & {j+1} (confidence: {conf:.2f}, reason: {reason})")
