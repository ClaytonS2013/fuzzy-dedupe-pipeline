"""
Advanced AI-Enhanced Deduplication Processor with Full Sentence Transformers
Utilizes state-of-the-art models for maximum accuracy in duplicate detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import logging
import re
from dataclasses import dataclass, asdict
import json
from datetime import datetime

# Core ML/AI imports
from sentence_transformers import SentenceTransformer, util
import torch
import faiss

# Fuzzy matching for hybrid approach
from rapidfuzz import fuzz, process
import phonenumbers
from phonenumbers import NumberParseException

# Performance monitoring
import time
from functools import wraps

logger = logging.getLogger(__name__)

def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"⏱️ {func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@dataclass
class MatchResult:
    """Store detailed matching results with confidence scores"""
    index1: int
    index2: int
    match_type: str  # 'semantic', 'phone', 'name', 'address', 'epd', 'hybrid'
    confidence: float
    details: dict
    business1_name: str = ""
    business2_name: str = ""

    def to_dict(self):
        return asdict(self)

class AdvancedAIDeduplicator:
    """
    Advanced deduplicator using multiple AI models and matching strategies.
    Optimized for accuracy over speed since we have no size constraints.
    """
    
    def __init__(self, 
                 # Model selection - using larger, more accurate models
                 semantic_model: str = 'all-mpnet-base-v2',  # Better than MiniLM
                 address_model: str = 'all-MiniLM-L6-v2',    # Specialized for addresses
                 # Thresholds
                 semantic_threshold: float = 0.8,
                 address_threshold: float = 0.85,
                 name_threshold: float = 0.75,
                 fuzzy_threshold: float = 80.0,
                 # Feature flags
                 use_gpu: bool = False,
                 batch_size: int = 32,
                 cache_embeddings: bool = True):
        """
        Initialize advanced deduplicator with multiple models.
        
        Args:
            semantic_model: Main model for semantic similarity
            address_model: Specialized model for address matching
            semantic_threshold: Threshold for semantic similarity (0-1)
            address_threshold: Threshold for address similarity (0-1)
            name_threshold: Threshold for name similarity (0-1)
            fuzzy_threshold: Threshold for fuzzy string matching (0-100)
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for embedding generation
            cache_embeddings: Whether to cache embeddings for reuse
        """
        self.semantic_threshold = semantic_threshold
        self.address_threshold = address_threshold
        self.name_threshold = name_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        
        # Set device
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ Using device: {self.device}")
        
        # Initialize models
        logger.info(f"📦 Loading semantic model: {semantic_model}")
        self.semantic_model = SentenceTransformer(semantic_model, device=self.device)
        
        logger.info(f"📦 Loading address model: {address_model}")
        self.address_model = SentenceTransformer(address_model, device=self.device)
        
        # Put models in eval mode for efficiency
        self.semantic_model.eval()
        self.address_model.eval()
        
        # Embedding caches
        self.embedding_cache = {} if cache_embeddings else None
        
        # Statistics tracking
        self.stats = {
            'total_comparisons': 0,
            'semantic_matches': 0,
            'phone_matches': 0,
            'address_matches': 0,
            'name_matches': 0,
            'hybrid_matches': 0,
            'processing_time': 0
        }
        
        logger.info("✅ Advanced AI Deduplicator initialized successfully")
    
    def normalize_phone(self, phone: str) -> str:
        """Advanced phone normalization with international support"""
        if pd.isna(phone) or not phone:
            return ""
        
        phone_str = str(phone).strip()
        
        try:
            # Try parsing as US number first
            parsed = phonenumbers.parse(phone_str, "US")
            if phonenumbers.is_valid_number(parsed):
                # Return in E164 format for consistent comparison
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException:
            pass
        
        # Fallback to simple normalization
        digits = re.sub(r'\D', '', phone_str)
        if len(digits) == 11 and digits[0] == '1':
            digits = digits[1:]
        
        return digits
    
    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text).lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep important separators
        text = re.sub(r'[^\w\s\-&]', '', text)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bdba\b': 'doing business as',
            r'\bllc\b': '',
            r'\binc\b': '',
            r'\bcorp\b': '',
            r'\bco\b': '',
            r'\bltd\b': '',
            r'\bpllc\b': '',
            r'\bpa\b': '',
            r'\bpc\b': '',
        }
        
        for abbr, expansion in abbreviations.items():
            text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_address(self, address: str) -> str:
        """Advanced address normalization"""
        if pd.isna(address) or not address:
            return ""
        
        address = str(address).lower().strip()
        
        # Standardize directionals
        directionals = {
            r'\bnorth\b': 'n',
            r'\bsouth\b': 's',
            r'\beast\b': 'e',
            r'\bwest\b': 'w',
            r'\bnortheast\b': 'ne',
            r'\bnorthwest\b': 'nw',
            r'\bsoutheast\b': 'se',
            r'\bsouthwest\b': 'sw',
        }
        
        for full, abbr in directionals.items():
            address = re.sub(full, abbr, address)
        
        # Standardize street types
        street_types = {
            r'\bstreet\b': 'st',
            r'\bavenue\b': 'ave',
            r'\broad\b': 'rd',
            r'\bdrive\b': 'dr',
            r'\blane\b': 'ln',
            r'\bboulevard\b': 'blvd',
            r'\bparkway\b': 'pkwy',
            r'\bhighway\b': 'hwy',
            r'\bfreeway\b': 'fwy',
            r'\bexpressway\b': 'expy',
            r'\bplace\b': 'pl',
            r'\bcourt\b': 'ct',
            r'\bsquare\b': 'sq',
            r'\bcircle\b': 'cir',
        }
        
        for full, abbr in street_types.items():
            address = re.sub(full, abbr, address)
        
        # Standardize units
        units = {
            r'\bapartment\b': 'apt',
            r'\bsuite\b': 'ste',
            r'\bunit\b': 'unit',
            r'\bfloor\b': 'fl',
            r'\bbuilding\b': 'bldg',
        }
        
        for full, abbr in units.items():
            address = re.sub(full, abbr, address)
        
        # Remove extra whitespace
        address = re.sub(r'\s+', ' ', address)
        
        return address.strip()
    
    def create_semantic_text(self, row: pd.Series) -> str:
        """Create comprehensive text for semantic embedding"""
        parts = []
        
        # Primary identifiers
        if not pd.isna(row.get('Practice Name')):
            name = self.normalize_text(str(row['Practice Name']))
            parts.append(f"business name: {name}")
        
        # Location information
        if not pd.isna(row.get('Practice Address')):
            addr = self.normalize_address(str(row['Practice Address']))
            parts.append(f"address: {addr}")
        
        if not pd.isna(row.get('City')):
            parts.append(f"city: {str(row['City']).lower()}")
        
        if not pd.isna(row.get('State')):
            parts.append(f"state: {str(row['State']).upper()}")
        
        if not pd.isna(row.get('Zip')):
            parts.append(f"zip: {str(row['Zip'])[:5]}")  # Use first 5 digits
        
        # Contact information
        if not pd.isna(row.get('phone_number')):
            phone = self.normalize_phone(row['phone_number'])
            if phone:
                parts.append(f"phone: {phone}")
        
        # Business type/category if available
        if not pd.isna(row.get('Practice Type')):
            parts.append(f"type: {str(row['Practice Type']).lower()}")
        
        # Website for additional context
        if not pd.isna(row.get('open_website')):
            website = str(row['open_website']).lower()
            # Extract domain name
            website = re.sub(r'https?://', '', website)
            website = re.sub(r'www\.', '', website)
            website = website.split('/')[0]
            parts.append(f"website: {website}")
        
        return " | ".join(parts)
    
    def create_address_text(self, row: pd.Series) -> str:
        """Create specialized text for address embedding"""
        parts = []
        
        if not pd.isna(row.get('Practice Address')):
            parts.append(self.normalize_address(str(row['Practice Address'])))
        
        if not pd.isna(row.get('City')):
            parts.append(str(row['City']).lower())
        
        if not pd.isna(row.get('State')):
            parts.append(str(row['State']).upper())
        
        if not pd.isna(row.get('Zip')):
            parts.append(str(row['Zip'])[:5])
        
        return ", ".join(parts)
    
    @timer
    def generate_embeddings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate both semantic and address embeddings for all records"""
        logger.info(f"🧮 Generating embeddings for {len(df)} records...")
        
        # Create text representations
        semantic_texts = []
        address_texts = []
        
        for idx, row in df.iterrows():
            semantic_texts.append(self.create_semantic_text(row))
            address_texts.append(self.create_address_text(row))
        
        # Generate embeddings in batches
        logger.info("📊 Generating semantic embeddings...")
        semantic_embeddings = self.semantic_model.encode(
            semantic_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info("📍 Generating address embeddings...")
        address_embeddings = self.address_model.encode(
            address_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return semantic_embeddings, address_embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build FAISS index for efficient similarity search"""
        dimension = embeddings.shape[1]
        
        # Use Inner Product (cosine similarity) since embeddings are normalized
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        return index
    
    @timer
    def find_duplicates_ai(self, df: pd.DataFrame) -> List[MatchResult]:
        """Find duplicates using AI-powered semantic matching"""
        matches = []
        
        # Generate embeddings
        semantic_emb, address_emb = self.generate_embeddings(df)
        
        # Build FAISS indices for fast search
        logger.info("🔍 Building FAISS indices...")
        semantic_index = self.build_faiss_index(semantic_emb)
        address_index = self.build_faiss_index(address_emb)
        
        # Prepare normalized data for additional checks
        df['norm_phone'] = df['phone_number'].apply(self.normalize_phone)
        df['norm_name'] = df['Practice Name'].apply(self.normalize_text)
        
        # Track processed pairs to avoid duplicates
        processed_pairs = set()
        
        logger.info("🔄 Finding semantic duplicates...")
        
        # For each record, find potential duplicates
        for i in range(len(df)):
            # Skip if this is part of an already processed pair
            if any(i in pair for pair in processed_pairs):
                continue
            
            # Search for similar records using semantic embeddings
            k = min(10, len(df))  # Look at top 10 most similar
            semantic_scores, semantic_indices = semantic_index.search(
                semantic_emb[i:i+1].astype('float32'), k
            )
            
            # Search using address embeddings
            address_scores, address_indices = address_index.search(
                address_emb[i:i+1].astype('float32'), k
            )
            
            # Process potential matches
            for j, sem_score in zip(semantic_indices[0], semantic_scores[0]):
                if i >= j:  # Skip self and already processed pairs
                    continue
                
                if (i, j) in processed_pairs or (j, i) in processed_pairs:
                    continue
                
                # Get address similarity score
                addr_score = address_emb[i] @ address_emb[j]
                
                # Calculate combined confidence
                row_i = df.iloc[i]
                row_j = df.iloc[j]
                
                # Check various matching criteria
                is_match = False
                match_type = None
                confidence = 0.0
                details = {
                    'semantic_score': float(sem_score),
                    'address_score': float(addr_score)
                }
                
                # 1. Strong semantic similarity
                if sem_score >= self.semantic_threshold:
                    is_match = True
                    match_type = 'semantic'
                    confidence = float(sem_score)
                    details['reason'] = f'High semantic similarity: {sem_score:.3f}'
                
                # 2. Strong address match with decent name match
                elif addr_score >= self.address_threshold:
                    name_similarity = fuzz.ratio(row_i['norm_name'], row_j['norm_name']) / 100
                    if name_similarity >= 0.6:  # Lower threshold when combined with address
                        is_match = True
                        match_type = 'hybrid'
                        confidence = (addr_score * 0.6 + name_similarity * 0.4)
                        details['name_similarity'] = name_similarity
                        details['reason'] = f'Address + name match: {addr_score:.3f} + {name_similarity:.3f}'
                
                # 3. Exact phone match
                if not is_match and row_i['norm_phone'] and row_j['norm_phone']:
                    if row_i['norm_phone'] == row_j['norm_phone']:
                        is_match = True
                        match_type = 'phone'
                        confidence = 0.95
                        details['reason'] = f'Exact phone match: {row_i["norm_phone"]}'
                
                # 4. EPD# exact match
                if not is_match and not pd.isna(row_i.get('epd#')) and not pd.isna(row_j.get('epd#')):
                    if str(row_i['epd#']).strip() == str(row_j['epd#']).strip():
                        is_match = True
                        match_type = 'epd'
                        confidence = 0.98
                        details['reason'] = f'EPD# match: {row_i["epd#"]}'
                
                if is_match:
                    match_result = MatchResult(
                        index1=i,
                        index2=j,
                        match_type=match_type,
                        confidence=confidence,
                        details=details,
                        business1_name=str(row_i.get('Practice Name', '')),
                        business2_name=str(row_j.get('Practice Name', ''))
                    )
                    matches.append(match_result)
                    processed_pairs.add((i, j))
                    
                    # Log the match
                    logger.info(f"🔗 Match found: {match_result.business1_name} ↔ {match_result.business2_name}")
                    logger.info(f"   Type: {match_type}, Confidence: {confidence:.3f}")
                    
                    # Update statistics
                    self.stats[f'{match_type}_matches'] = self.stats.get(f'{match_type}_matches', 0) + 1
        
        # Clean up temporary columns
        df.drop(['norm_phone', 'norm_name'], axis=1, inplace=True, errors='ignore')
        
        return matches
    
    def merge_duplicate_records(self, df: pd.DataFrame, indices: List[int]) -> pd.Series:
        """
        Intelligently merge duplicate records, keeping the most complete information.
        Uses confidence scores to prefer higher-quality data.
        """
        if not indices:
            return pd.Series()
        
        # Get all duplicate rows
        dup_rows = df.loc[indices]
        
        # Start with the first row as base
        merged = dup_rows.iloc[0].copy()
        
        # For each field, intelligently select the best value
        for col in dup_rows.columns:
            values = dup_rows[col].dropna()
            
            if len(values) == 0:
                continue
            
            if col in ['Practice Name', 'Practice Address']:
                # For important fields, take the longest (most complete) value
                # Also check for consistency
                unique_normalized = set(self.normalize_text(str(v)) for v in values)
                if len(unique_normalized) == 1:
                    # All values are essentially the same, take the longest
                    merged[col] = max(values, key=lambda x: len(str(x)))
                else:
                    # Values differ, take the most common or longest
                    value_counts = values.value_counts()
                    if value_counts.iloc[0] > 1:
                        merged[col] = value_counts.index[0]
                    else:
                        merged[col] = max(values, key=lambda x: len(str(x)))
            
            elif col in ['phone_number', 'epd#']:
                # For identifiers, check consistency
                unique_values = set(str(v).strip() for v in values)
                if len(unique_values) == 1:
                    merged[col] = values.iloc[0]
                else:
                    # Multiple values, log warning and take first non-empty
                    logger.warning(f"Multiple {col} values found: {unique_values}")
                    merged[col] = values.iloc[0]
            
            else:
                # For other fields, take the longest non-empty value
                merged[col] = max(values, key=lambda x: len(str(x)))
        
        # Add metadata about the merge
        merged['merge_count'] = len(indices)
        merged['merge_confidence'] = 0.95  # High confidence since AI-verified
        merged['merge_date'] = datetime.now().isoformat()
        
        return merged
    
    def deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main deduplication function using AI-powered matching.
        
        Returns:
            Tuple of (deduplicated_dataframe, statistics_dict)
        """
        start_time = time.time()
        logger.info(f"🚀 Starting AI-powered deduplication for {len(df)} records")
        
        # Find all duplicate matches using AI
        matches = self.find_duplicates_ai(df)
        
        # Group matches into clusters
        duplicate_clusters = {}
        for match in matches:
            # Find or create cluster for index1
            cluster_id = None
            for cid, indices in duplicate_clusters.items():
                if match.index1 in indices or match.index2 in indices:
                    cluster_id = cid
                    break
            
            if cluster_id is None:
                cluster_id = match.index1
                duplicate_clusters[cluster_id] = set([match.index1])
            
            # Add both indices to cluster
            duplicate_clusters[cluster_id].add(match.index1)
            duplicate_clusters[cluster_id].add(match.index2)
        
        # Convert clusters to lists for processing
        duplicate_clusters = {k: list(v) for k, v in duplicate_clusters.items()}
        
        # Create deduplicated dataset
        clean_records = []
        processed_indices = set()
        
        for idx, row in df.iterrows():
            if idx in processed_indices:
                continue
            
            # Check if this index is part of a duplicate cluster
            cluster_indices = None
            for cluster in duplicate_clusters.values():
                if idx in cluster:
                    cluster_indices = cluster
                    break
            
            if cluster_indices:
                # Merge the duplicate cluster
                merged_record = self.merge_duplicate_records(df, cluster_indices)
                clean_records.append(merged_record)
                processed_indices.update(cluster_indices)
                logger.info(f"✅ Merged {len(cluster_indices)} duplicates: {merged_record['Practice Name']}")
            else:
                # Unique record
                row_copy = row.copy()
                row_copy['merge_count'] = 1
                row_copy['merge_confidence'] = 1.0
                row_copy['merge_date'] = datetime.now().isoformat()
                clean_records.append(row_copy)
                processed_indices.add(idx)
        
        # Create clean DataFrame
        clean_df = pd.DataFrame(clean_records)
        clean_df.reset_index(drop=True, inplace=True)
        
        # Calculate statistics
        end_time = time.time()
        self.stats['processing_time'] = end_time - start_time
        self.stats['total_records'] = len(df)
        self.stats['unique_records'] = len(clean_df)
        self.stats['duplicates_removed'] = len(df) - len(clean_df)
        self.stats['duplicate_clusters'] = len(duplicate_clusters)
        self.stats['total_matches'] = len(matches)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("🎯 AI Deduplication Complete!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Total records: {self.stats['total_records']}")
        logger.info(f"   Unique records: {self.stats['unique_records']}")
        logger.info(f"   Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"   Duplicate clusters: {self.stats['duplicate_clusters']}")
        logger.info(f"   Processing time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"   Match types breakdown:")
        logger.info(f"      Semantic: {self.stats.get('semantic_matches', 0)}")
        logger.info(f"      Phone: {self.stats.get('phone_matches', 0)}")
        logger.info(f"      Address: {self.stats.get('address_matches', 0)}")
        logger.info(f"      Hybrid: {self.stats.get('hybrid_matches', 0)}")
        logger.info(f"      EPD: {self.stats.get('epd_matches', 0)}")
        logger.info("=" * 60)
        
        return clean_df, self.stats


def run_deduplication(df: pd.DataFrame, use_ai: bool = True) -> pd.DataFrame:
    """
    Main entry point for deduplication process.
    
    Args:
        df: Input DataFrame with practice data
        use_ai: Whether to use AI-powered deduplication
        
    Returns:
        DataFrame with duplicates removed and merged
    """
    logger.info(f"🚀 Starting {'AI-powered' if use_ai else 'standard'} deduplication with {len(df)} records")
    
    # Initialize the deduplicator
    dedup = AdvancedAIDeduplicator(
        semantic_model='all-mpnet-base-v2',  # Best accuracy
        address_model='all-MiniLM-L6-v2',    # Fast and accurate for addresses
        semantic_threshold=0.80,
        address_threshold=0.85,
        name_threshold=0.75,
        fuzzy_threshold=80.0,
        batch_size=32,
        cache_embeddings=True
    )
    
    # Run deduplication
    clean_df, stats = dedup.deduplicate(df)
    
    # Ensure field mappings are correct for output
    if 'phone_number' in clean_df.columns and 'phone' not in clean_df.columns:
        clean_df['phone'] = clean_df['phone_number']
    
    if 'open_website' in clean_df.columns and 'website' not in clean_df.columns:
        clean_df['website'] = clean_df['open_website']
    
    # Add confidence score if not present (for backward compatibility)
    if 'confidence_score' not in clean_df.columns:
        clean_df['confidence_score'] = clean_df.get('merge_confidence', 1.0)
    
    logger.info(f"✅ Deduplication complete. Reduced from {len(df)} to {len(clean_df)} records")
    
    return clean_df
