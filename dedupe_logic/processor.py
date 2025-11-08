"""
Deduplication Processor Module
Handles the core deduplication logic for the pipeline
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

def run_deduplication(supabase_client) -> int:
    """
    Main deduplication function that processes records from practice_records table
    
    Args:
        supabase_client: Initialized Supabase client
    
    Returns:
        int: Number of unique records after deduplication
    """
    try:
        logger.info("🔍 Starting deduplication process...")
        
        # Fetch all records from practice_records
        logger.info("📥 Fetching records from practice_records...")
        result = supabase_client.table('practice_records').select("*").execute()
        
        if not result.data:
            logger.warning("⚠️ No records found in practice_records table")
            return 0
            
        records = result.data
        logger.info(f"📊 Found {len(records)} records to process")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(records)
        
        # Perform deduplication
        dedupe_results = deduplicate_records(df)
        
        # Save results to dedupe_results table
        if dedupe_results:
            logger.info(f"💾 Saving {len(dedupe_results)} unique records to dedupe_results...")
            
            # Clear existing results
            supabase_client.table('dedupe_results').delete().neq('id', 0).execute()
            
            # Insert new results
            supabase_client.table('dedupe_results').insert(dedupe_results).execute()
            logger.info("✅ Deduplication results saved successfully")
            
        return len(dedupe_results)
        
    except Exception as e:
        logger.error(f"❌ Deduplication failed: {str(e)}")
        raise


def deduplicate_records(df: pd.DataFrame) -> List[Dict]:
    """
    Core deduplication logic using multiple matching strategies
    
    Args:
        df: DataFrame containing records to deduplicate
    
    Returns:
        List of deduplicated records
    """
    logger.info("🔄 Processing duplicates...")
    
    # Initialize cluster tracking
    cluster_id = 1
    df['cluster_id'] = None
    df['confidence_score'] = 100.0
    df['duplicate_count'] = 1
    
    # Strategy 1: Exact name matching (case-insensitive)
    df = apply_name_matching(df, cluster_id)
    cluster_id = df['cluster_id'].max() + 1 if df['cluster_id'].notna().any() else cluster_id
    
    # Strategy 2: Address matching
    df = apply_address_matching(df, cluster_id)
    cluster_id = df['cluster_id'].max() + 1 if df['cluster_id'].notna().any() else cluster_id
    
    # Strategy 3: Phone number matching
    df = apply_phone_matching(df, cluster_id)
    cluster_id = df['cluster_id'].max() + 1 if df['cluster_id'].notna().any() else cluster_id
    
    # Strategy 4: Email domain matching
    df = apply_email_matching(df, cluster_id)
    
    # Group by cluster and merge
    merged_records = merge_duplicate_clusters(df)
    
    logger.info(f"✅ Reduced {len(df)} records to {len(merged_records)} unique records")
    
    return merged_records


def apply_name_matching(df: pd.DataFrame, start_cluster: int) -> pd.DataFrame:
    """Apply name-based duplicate detection"""
    cluster_id = start_cluster
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('cluster_id')):
            continue
            
        name = str(row.get('name', '')).lower().strip()
        if not name:
            continue
            
        # Find similar names
        for idx2, row2 in df.iterrows():
            if idx == idx2 or pd.notna(row2.get('cluster_id')):
                continue
                
            name2 = str(row2.get('name', '')).lower().strip()
            
            # Check for exact match or common variations
            if names_are_similar(name, name2):
                if pd.isna(row.get('cluster_id')):
                    df.at[idx, 'cluster_id'] = f'cluster_{cluster_id}'
                    df.at[idx, 'confidence_score'] = 95.0
                
                df.at[idx2, 'cluster_id'] = f'cluster_{cluster_id}'
                df.at[idx2, 'confidence_score'] = 95.0
        
        if pd.notna(df.at[idx, 'cluster_id']):
            cluster_id += 1
    
    return df


def apply_address_matching(df: pd.DataFrame, start_cluster: int) -> pd.DataFrame:
    """Apply address-based duplicate detection"""
    cluster_id = start_cluster
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('cluster_id')):
            continue
            
        address = normalize_address(str(row.get('address', '')))
        if not address:
            continue
            
        for idx2, row2 in df.iterrows():
            if idx == idx2 or pd.notna(row2.get('cluster_id')):
                continue
                
            address2 = normalize_address(str(row2.get('address', '')))
            
            if address == address2:
                if pd.isna(row.get('cluster_id')):
                    df.at[idx, 'cluster_id'] = f'cluster_{cluster_id}'
                    df.at[idx, 'confidence_score'] = 90.0
                
                df.at[idx2, 'cluster_id'] = f'cluster_{cluster_id}'
                df.at[idx2, 'confidence_score'] = 90.0
        
        if pd.notna(df.at[idx, 'cluster_id']):
            cluster_id += 1
    
    return df


def apply_phone_matching(df: pd.DataFrame, start_cluster: int) -> pd.DataFrame:
    """Apply phone number-based duplicate detection"""
    cluster_id = start_cluster
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('cluster_id')):
            continue
            
        phone = normalize_phone(str(row.get('phone', '')))
        if not phone or len(phone) < 10:
            continue
            
        for idx2, row2 in df.iterrows():
            if idx == idx2 or pd.notna(row2.get('cluster_id')):
                continue
                
            phone2 = normalize_phone(str(row2.get('phone', '')))
            
            if phone == phone2:
                if pd.isna(row.get('cluster_id')):
                    df.at[idx, 'cluster_id'] = f'cluster_{cluster_id}'
                    df.at[idx, 'confidence_score'] = 98.0
                
                df.at[idx2, 'cluster_id'] = f'cluster_{cluster_id}'
                df.at[idx2, 'confidence_score'] = 98.0
        
        if pd.notna(df.at[idx, 'cluster_id']):
            cluster_id += 1
    
    return df


def apply_email_matching(df: pd.DataFrame, start_cluster: int) -> pd.DataFrame:
    """Apply email-based duplicate detection"""
    cluster_id = start_cluster
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('cluster_id')):
            continue
            
        email = str(row.get('email', '')).lower().strip()
        if not email or '@' not in email:
            continue
            
        for idx2, row2 in df.iterrows():
            if idx == idx2 or pd.notna(row2.get('cluster_id')):
                continue
                
            email2 = str(row2.get('email', '')).lower().strip()
            
            if email == email2:
                if pd.isna(row.get('cluster_id')):
                    df.at[idx, 'cluster_id'] = f'cluster_{cluster_id}'
                    df.at[idx, 'confidence_score'] = 99.0
                
                df.at[idx2, 'cluster_id'] = f'cluster_{cluster_id}'
                df.at[idx2, 'confidence_score'] = 99.0
        
        if pd.notna(df.at[idx, 'cluster_id']):
            cluster_id += 1
    
    return df


def merge_duplicate_clusters(df: pd.DataFrame) -> List[Dict]:
    """
    Merge records within the same cluster, keeping the most complete information
    """
    merged_records = []
    
    # Process clustered records
    clustered = df[df['cluster_id'].notna()]
    for cluster_id in clustered['cluster_id'].unique():
        cluster_records = clustered[clustered['cluster_id'] == cluster_id]
        merged_record = merge_cluster_records(cluster_records)
        merged_record['cluster_id'] = cluster_id
        merged_record['duplicate_count'] = len(cluster_records)
        merged_records.append(merged_record)
    
    # Add non-clustered records (unique records)
    unique_records = df[df['cluster_id'].isna()]
    for _, record in unique_records.iterrows():
        record_dict = record.to_dict()
        record_dict['cluster_id'] = f'unique_{len(merged_records) + 1}'
        record_dict['duplicate_count'] = 1
        record_dict['confidence_score'] = 100.0
        merged_records.append(clean_record(record_dict))
    
    return merged_records


def merge_cluster_records(cluster_df: pd.DataFrame) -> Dict:
    """
    Merge multiple records from the same cluster into one
    Priority: Keep the most complete and recent information
    """
    merged = {}
    
    # Priority fields (take first non-empty)
    priority_fields = ['id', 'name', 'address', 'city', 'state', 'zip', 
                      'phone', 'email', 'website', 'place_id']
    
    for field in priority_fields:
        for _, row in cluster_df.iterrows():
            value = row.get(field)
            if value and str(value).strip() and str(value).lower() not in ['nan', 'none']:
                merged[field] = value
                break
    
    # Average confidence score
    merged['confidence_score'] = cluster_df['confidence_score'].mean()
    
    return clean_record(merged)


def clean_record(record: Dict) -> Dict:
    """Clean and format a record for output"""
    cleaned = {}
    
    # Define the fields we want to keep in the output
    output_fields = ['id', 'name', 'address', 'city', 'state', 'zip',
                    'phone', 'email', 'website', 'cluster_id', 
                    'confidence_score', 'duplicate_count']
    
    for field in output_fields:
        value = record.get(field)
        if value is not None and str(value).lower() not in ['nan', 'none', '']:
            cleaned[field] = value
        else:
            cleaned[field] = None
    
    return cleaned


def names_are_similar(name1: str, name2: str) -> bool:
    """Check if two names are similar enough to be duplicates"""
    # Exact match
    if name1 == name2:
        return True
    
    # Check if one is contained in the other
    if name1 in name2 or name2 in name1:
        return True
    
    # Check for common variations
    variations = [
        ('test', ''),  # Remove 'test' suffix for test data
        ('inc', 'incorporated'),
        ('corp', 'corporation'),
        ('llc', ''),
        ('ltd', 'limited'),
        ('.', ''),
        (',', ''),
        ('&', 'and')
    ]
    
    clean1 = name1
    clean2 = name2
    
    for old, new in variations:
        clean1 = clean1.replace(old, new)
        clean2 = clean2.replace(old, new)
    
    clean1 = ' '.join(clean1.split())
    clean2 = ' '.join(clean2.split())
    
    return clean1 == clean2


def normalize_address(address: str) -> str:
    """Normalize address for comparison"""
    if not address:
        return ''
    
    address = address.lower().strip()
    
    # Common replacements
    replacements = {
        'street': 'st',
        'avenue': 'ave',
        'road': 'rd',
        'drive': 'dr',
        'lane': 'ln',
        'court': 'ct',
        'place': 'pl',
        'boulevard': 'blvd',
        'north': 'n',
        'south': 's',
        'east': 'e',
        'west': 'w'
    }
    
    for old, new in replacements.items():
        address = address.replace(old, new)
    
    # Remove extra spaces
    address = ' '.join(address.split())
    
    return address


def normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison"""
    if not phone:
        return ''
    
    # Remove all non-numeric characters
    phone = ''.join(c for c in phone if c.isdigit())
    
    # Remove country code if present (1 for US)
    if len(phone) == 11 and phone.startswith('1'):
        phone = phone[1:]
    
    return phone


# Main entry point for backwards compatibility
def process_duplicates(records: List[Dict]) -> List[Dict]:
    """Legacy function for compatibility"""
    df = pd.DataFrame(records)
    return deduplicate_records(df)
