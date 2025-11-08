"""
Advanced AI-powered deduplication processor with COMPLETE field mapping fix
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import AI libraries
AI_AVAILABLE = False
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    AI_AVAILABLE = True
    logger.info("✅ AI libraries imported successfully")
except ImportError as e:
    logger.warning(f"AI libraries not available: {e}")
    logger.warning("Falling back to rule-based deduplication")


class AdvancedAIDeduplicator:
    """Advanced deduplicator using AI semantic matching."""
    
    def __init__(self, semantic_threshold=0.8, address_threshold=0.85):
        """Initialize AI deduplicator with models."""
        self.semantic_threshold = semantic_threshold
        self.address_threshold = address_threshold
        
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ Using device: {self.device}")
        
        # Load models
        logger.info("📦 Loading semantic model: all-mpnet-base-v2")
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        
        logger.info("📦 Loading address model: all-MiniLM-L6-v2")
        self.address_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        logger.info("✅ Advanced AI Deduplicator initialized successfully")
    
    def normalize_phone(self, phone: str) -> str:
        """Normalize phone numbers to digits only."""
        if not phone or phone == '#ERROR!':
            return ''
        return re.sub(r'\D', '', str(phone))
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ''
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def generate_embeddings(self, records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for semantic and address matching."""
        start_time = time.time()
        
        # Prepare texts for embedding
        semantic_texts = []
        address_texts = []
        
        for record in records:
            # Semantic text (name + category)
            name = record.get('name', '')
            category = record.get('category', '')
            semantic_text = f"{name} {category}".strip()
            semantic_texts.append(semantic_text if semantic_text else "unknown")
            
            # Address text
            address_parts = [
                str(record.get('address', '')),
                str(record.get('Street Address', '')),
                str(record.get('City', '')),
                str(record.get('State', '')),
                str(record.get('Zip', ''))
            ]
            address_text = ' '.join(filter(None, address_parts))
            address_texts.append(address_text if address_text else "unknown")
        
        # Generate embeddings
        logger.info("📊 Generating semantic embeddings...")
        semantic_embeddings = self.semantic_model.encode(
            semantic_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        logger.info("📍 Generating address embeddings...")
        address_embeddings = self.address_model.encode(
            address_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ generate_embeddings took {elapsed:.2f} seconds")
        
        return semantic_embeddings, address_embeddings
    
    def find_duplicates_ai(self, records: List[Dict]) -> List[Dict]:
        """Find duplicates using AI-powered matching."""
        start_time = time.time()
        
        if len(records) <= 1:
            return []
        
        logger.info(f"🧮 Generating embeddings for {len(records)} records...")
        semantic_embeddings, address_embeddings = self.generate_embeddings(records)
        
        # Build FAISS indices for fast similarity search
        logger.info("🔍 Building FAISS indices...")
        semantic_index = faiss.IndexFlatIP(semantic_embeddings.shape[1])
        address_index = faiss.IndexFlatIP(address_embeddings.shape[1])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(semantic_embeddings)
        faiss.normalize_L2(address_embeddings)
        
        semantic_index.add(semantic_embeddings)
        address_index.add(address_embeddings)
        
        # Find duplicates
        duplicates = []
        processed_pairs = set()
        
        logger.info("🔄 Finding semantic duplicates...")
        k = min(10, len(records))
        semantic_distances, semantic_indices = semantic_index.search(semantic_embeddings, k)
        
        for i in range(len(records)):
            for j_idx, score in zip(semantic_indices[i], semantic_distances[i]):
                if i >= j_idx:
                    continue
                    
                pair = (i, j_idx)
                if pair in processed_pairs:
                    continue
                    
                if score >= self.semantic_threshold:
                    processed_pairs.add(pair)
                    
                    # Check address similarity
                    address_score = np.dot(address_embeddings[i], address_embeddings[j_idx])
                    
                    # Check phone similarity
                    phone1 = self.normalize_phone(records[i].get('phone_number', ''))
                    phone2 = self.normalize_phone(records[j_idx].get('phone_number', ''))
                    phone_match = phone1 and phone2 and phone1 == phone2
                    
                    # Determine match type
                    if phone_match and address_score >= 0.7:
                        match_type = 'hybrid'
                        confidence = min(0.95, (score + address_score) / 2)
                    elif address_score >= self.address_threshold:
                        match_type = 'semantic+address'
                        confidence = min(0.95, (score + address_score) / 2)
                    else:
                        match_type = 'semantic'
                        confidence = score
                    
                    duplicate = {
                        'record1_id': records[i].get('id'),
                        'record2_id': records[j_idx].get('id'),
                        'record1_name': records[i].get('name'),
                        'record2_name': records[j_idx].get('name'),
                        'match_type': match_type,
                        'confidence': float(confidence),
                        'semantic_score': float(score),
                        'address_score': float(address_score),
                        'phone_match': phone_match
                    }
                    
                    duplicates.append(duplicate)
                    
                    logger.info(f"🔗 Match found: {duplicate['record1_name']} ↔ {duplicate['record2_name']}")
                    logger.info(f"   Type: {match_type}, Confidence: {confidence:.3f}")
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ find_duplicates_ai took {elapsed:.2f} seconds")
        
        return duplicates
    
    def cluster_duplicates(self, records: List[Dict], duplicates: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Cluster duplicates and return unique records with cluster information."""
        # Build adjacency list
        adjacency = defaultdict(set)
        confidence_map = {}
        
        for dup in duplicates:
            id1, id2 = dup['record1_id'], dup['record2_id']
            adjacency[id1].add(id2)
            adjacency[id2].add(id1)
            confidence_map[(id1, id2)] = dup['confidence']
            confidence_map[(id2, id1)] = dup['confidence']
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for record in records:
            record_id = record['id']
            if record_id not in visited:
                # BFS to find cluster
                cluster = set()
                queue = [record_id]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    cluster.add(current)
                    
                    for neighbor in adjacency.get(current, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                clusters.append(cluster)
        
        # Select representative from each cluster
        unique_records = []
        cluster_info = {}
        
        for cluster_idx, cluster in enumerate(clusters):
            # Get all records in cluster
            cluster_records = [r for r in records if r['id'] in cluster]
            
            # Select representative (prefer most complete record)
            representative = max(cluster_records, key=lambda r: (
                len(str(r.get('name', ''))),
                len(str(r.get('address', ''))),
                len(str(r.get('phone_number', ''))),
                r.get('reviews_count', 0)
            ))
            
            # Track cluster information
            for record_id in cluster:
                cluster_info[record_id] = {
                    'cluster_id': representative['id'],
                    'duplicate_count': len(cluster),
                    'confidence': max([confidence_map.get((record_id, other), 0) 
                                     for other in cluster if other != record_id] or [1.0])
                }
            
            unique_records.append(representative)
            
            if len(cluster) > 1:
                logger.info(f"✅ Merged {len(cluster)-1} duplicates: {representative.get('name')}")
        
        return unique_records, cluster_info


def run_deduplication(supabase_client):
    """Run deduplication process with COMPLETE field mapping fix."""
    try:
        # Fetch all practice records
        response = supabase_client.table('practice_records').select("*").execute()
        all_records = response.data
        
        if not all_records:
            logger.warning("No records found in practice_records")
            return 0
        
        logger.info(f"🚀 Starting deduplication with {len(all_records)} records")
        
        # Initialize deduplicator
        use_ai = True
        if use_ai and AI_AVAILABLE:
            try:
                logger.info("🤖 Attempting to initialize AI deduplicator...")
                deduplicator = AdvancedAIDeduplicator()
                logger.info("✅ AI deduplicator initialized successfully")
                
                # Run AI deduplication
                logger.info(f"🚀 Starting AI-powered deduplication for {len(all_records)} records")
                duplicates = deduplicator.find_duplicates_ai(all_records)
                unique_records, cluster_info = deduplicator.cluster_duplicates(all_records, duplicates)
                
            except Exception as e:
                logger.error(f"AI deduplication failed: {e}")
                raise
        else:
            logger.warning("AI deduplication not available")
            unique_records = all_records
            cluster_info = {r['id']: {'cluster_id': r['id'], 'duplicate_count': 1, 'confidence': 1.0} 
                           for r in all_records}
        
        # Print statistics
        logger.info("=" * 60)
        logger.info("🎯 AI Deduplication Complete!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Total records: {len(all_records)}")
        logger.info(f"   Unique records: {len(unique_records)}")
        logger.info(f"   Duplicates removed: {len(all_records) - len(unique_records)}")
        logger.info("=" * 60)
        
        logger.info(f"✅ AI deduplication complete. Reduced from {len(all_records)} to {len(unique_records)} records")
        
        # COMPLETE FIX: Prepare dedupe results with proper field extraction and mapping
        dedupe_results = []
        for record in unique_records:
            record_id = record.get('id')
            record_cluster_info = cluster_info.get(record_id, {})
            
            # Debug: Log first record's fields to understand structure
            if len(dedupe_results) == 0:
                logger.info(f"📋 Available fields in first record: {list(record.keys())[:20]}")  # Show first 20 fields
            
            # Extract city, state, zip with better logic
            city_value = ''
            state_value = ''
            zip_value = ''
            
            # Try to get city, state, zip from individual fields first (with capital letters)
            city_value = str(record.get('City', '') or record.get('city', '') or '').strip()
            state_value = str(record.get('State', '') or record.get('state', '') or '').strip()
            zip_value = str(record.get('Zip', '') or record.get('zip', '') or '').strip()
            
            # Clean up bad values
            if zip_value.lower() in ['none', 'nan', 'null', '']:
                zip_value = ''
            if city_value.lower() in ['none', 'nan', 'null']:
                city_value = ''
            if state_value.lower() in ['none', 'nan', 'null']:
                state_value = ''
            
            # If any are missing, try to extract from address
            if not all([city_value, state_value, zip_value]):
                # Try Full Address or address field
                full_addr = record.get('Full Address', '') or record.get('address', '') or ''
                
                if full_addr:
                    # Extract ZIP code using regex
                    if not zip_value:
                        zip_match = re.search(r'\b(\d{5})(?:-\d{4})?\b', str(full_addr))
                        if zip_match:
                            zip_value = zip_match.group(1)
                    
                    # Try to parse city and state from address
                    # Format usually: "Street, City, State ZIP"
                    addr_parts = str(full_addr).split(',')
                    if len(addr_parts) >= 3:
                        # Extract city if not already found
                        if not city_value and len(addr_parts) >= 2:
                            potential_city = addr_parts[-2].strip()
                            if potential_city and not any(char.isdigit() for char in potential_city[:3]):
                                city_value = potential_city
                        
                        # Extract state if not already found
                        if not state_value and len(addr_parts) >= 1:
                            last_part = addr_parts[-1].strip()
                            # Look for state abbreviation (2 capital letters)
                            state_match = re.search(r'\b([A-Z]{2})\b', last_part)
                            if state_match:
                                state_value = state_match.group(1)
            
            # Build the cleaned result record
            result = {
                'id': record_id,
                
                # Name - straightforward
                'name': str(record.get('name', '') or '').strip(),
                
                # Address - try multiple field names
                'address': str(
                    record.get('address') or 
                    record.get('Street Address') or 
                    record.get('Full Address') or 
                    ''
                ).strip(),
                
                # Geographic fields with extraction logic
                'city': city_value,
                'state': state_value,
                'zip': zip_value,
                
                # Phone - note the field name is phone_number in source
                'phone': str(
                    record.get('phone_number') or 
                    record.get('phone') or 
                    ''
                ).strip(),
                
                # Email - likely doesn't exist in veterinary data but check anyway
                'email': str(
                    record.get('email') or 
                    record.get('Email') or 
                    record.get('email_address') or 
                    ''
                ).strip(),
                
                # Website - field name is open_website in source
                'website': str(
                    record.get('open_website') or 
                    record.get('website') or 
                    record.get('url') or 
                    ''
                ).strip(),
                
                # Cluster information
                'cluster_id': record_cluster_info.get('cluster_id', record_id),
                'confidence_score': float(record_cluster_info.get('confidence', 1.0)),
                'duplicate_count': int(record_cluster_info.get('duplicate_count', 1))
            }
            
            # Final cleanup - remove any remaining 'None' string values
            for key, value in result.items():
                if isinstance(value, str):
                    # Remove None/nan/null strings
                    if value.lower() in ['none', 'nan', 'null']:
                        result[key] = ''
                    # Remove #ERROR! values
                    elif value == '#ERROR!':
                        result[key] = ''
            
            dedupe_results.append(result)
        
        # Log sample record after mapping to verify
        if dedupe_results:
            sample = dedupe_results[0]
            logger.info(f"📊 Sample record after complete field mapping:")
            logger.info(f"  ID: {sample['id']}")
            logger.info(f"  Name: {sample['name']}")
            logger.info(f"  Address: {sample['address']}")
            logger.info(f"  City: {sample['city']}")
            logger.info(f"  State: {sample['state']}")
            logger.info(f"  Zip: {sample['zip']}")
            logger.info(f"  Phone: {sample['phone']}")
            logger.info(f"  Email: {sample['email']}")
            logger.info(f"  Website: {sample['website']}")
            logger.info(f"  Cluster ID: {sample['cluster_id']}")
            logger.info(f"  Confidence: {sample['confidence_score']}")
            logger.info(f"  Duplicate Count: {sample['duplicate_count']}")
        
        # Clear existing results
        logger.info("🗑️ Clearing existing dedupe_results...")
        supabase_client.table('dedupe_results').delete().neq('id', 0).execute()
        
        # Insert new results with all fields
        if dedupe_results:
            logger.info(f"📝 Inserting {len(dedupe_results)} deduplicated records...")
            
            # Insert in batches if needed
            batch_size = 50
            for i in range(0, len(dedupe_results), batch_size):
                batch = dedupe_results[i:i+batch_size]
                try:
                    supabase_client.table('dedupe_results').insert(batch).execute()
                    logger.info(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} records")
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    logger.error(f"First record in failed batch: {batch[0] if batch else 'empty batch'}")
                    raise
        
        logger.info(f"✅ Successfully saved {len(dedupe_results)} deduplicated records")
        return len(unique_records)
        
    except Exception as e:
        logger.error(f"❌ Error in deduplication: {str(e)}")
        raise
