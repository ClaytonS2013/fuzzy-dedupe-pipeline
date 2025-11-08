"""
AI-Enhanced Fuzzy Matching Processor for Veterinary Practice Deduplication
Now with Claude Sonnet 3.5 for intelligent validation
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

# [Previous imports remain the same...]

# Replace the OpenAI import with Anthropic
try:
    from anthropic import Anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ Anthropic not installed. Run: pip install anthropic")

# ============================================
# PART 4: CLAUDE SONNET 3.5 VALIDATION
# ============================================

class ClaudeLLMValidator:
    """Claude Sonnet 3.5 validation for complex deduplication cases"""
    
    def __init__(self):
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        if self.api_key and LLM_AVAILABLE:
            self.client = Anthropic(api_key=self.api_key)
            self.enabled = True
            self.model = "claude-3-5-sonnet-20241022"  # Latest Sonnet 3.5
            logger.info("✅ Claude Sonnet 3.5 validation enabled")
        else:
            self.enabled = False
            logger.info("ℹ️ Claude validation not configured (set ANTHROPIC_API_KEY)")
    
    def validate_match(self, record1: dict, record2: dict, current_score: float) -> dict:
        """Use Claude to validate a potential match with nuanced understanding"""
        if not self.enabled:
            return {'use_llm': False}
        
        # Only use Claude for uncertain cases (0.5 to 0.85 confidence)
        if current_score < 0.5 or current_score > 0.85:
            return {'use_llm': False}
        
        # Create a detailed prompt for Claude
        prompt = f"""You are an expert at identifying duplicate veterinary practice records. Analyze these two records carefully:

Practice 1:
- Name: {record1.get('practice_name', 'Not provided')}
- Phone: {record1.get('phone', 'Not provided')}
- Address: {record1.get('address', 'Not provided')}
- City: {record1.get('city', 'Not provided')}
- State: {record1.get('state', 'Not provided')}
- Email: {record1.get('email', 'Not provided')}
- Website: {record1.get('website', 'Not provided')}

Practice 2:
- Name: {record2.get('practice_name', 'Not provided')}
- Phone: {record2.get('phone', 'Not provided')}
- Address: {record2.get('address', 'Not provided')}
- City: {record2.get('city', 'Not provided')}
- State: {record2.get('state', 'Not provided')}
- Email: {record2.get('email', 'Not provided')}
- Website: {record2.get('website', 'Not provided')}

Current fuzzy match score: {current_score:.2%}

Consider these factors:
1. Common veterinary practice name variations (e.g., "Vet" vs "Veterinary", "Animal Hospital" vs "Pet Clinic")
2. Chain locations or franchises that share names but are different locations
3. Practice acquisitions or name changes
4. Common abbreviations and acronyms
5. Parent/subsidiary relationships
6. Mobile vs physical locations
7. Emergency vs regular practice locations

Respond with ONLY a valid JSON object (no markdown, no explanation outside JSON):
{{"is_duplicate": true_or_false, "confidence": 0.0_to_1.0, "relationship": "same_entity|different_locations|parent_subsidiary|completely_different|name_change", "reasoning": "brief explanation"}}"""
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistency
                system="You are a data deduplication expert specializing in veterinary practice records. Always respond with valid JSON only.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            response_text = response.content[0].text.strip()
            
            # Clean up response if it has markdown
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            result['use_llm'] = True
            result['model'] = 'claude-3.5-sonnet'
            
            logger.info(f"🤖 Claude validation: {result['relationship']} (confidence: {result['confidence']:.2%})")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Claude response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {'use_llm': False, 'error': 'json_parse_error'}
        except Exception as e:
            logger.warning(f"Claude validation failed: {e}")
            return {'use_llm': False, 'error': str(e)}
    
    def validate_batch(self, matches: List[dict], max_validations: int = 20) -> List[dict]:
        """Validate multiple matches efficiently with rate limiting"""
        if not self.enabled:
            return matches
        
        validated_count = 0
        
        for match in matches:
            # Only validate uncertain matches
            if 0.5 < match.get('combined_score', 0) < 0.85:
                if validated_count >= max_validations:
                    logger.info(f"⚠️ Reached max validations limit ({max_validations})")
                    break
                
                llm_result = self.validate_match(
                    match['record1'],
                    match['record2'],
                    match.get('combined_score', 0)
                )
                
                if llm_result.get('use_llm'):
                    match['llm_validation'] = llm_result
                    match['llm_confidence'] = llm_result['confidence']
                    match['llm_reasoning'] = llm_result.get('reasoning', '')
                    match['relationship_type'] = llm_result.get('relationship', 'unknown')
                    validated_count += 1
        
        logger.info(f"✅ Validated {validated_count} matches with Claude")
        return matches

# ============================================
# ENHANCED MAIN DEDUPLICATOR CLASS
# ============================================

class AIEnhancedDeduplicator:
    """Main class that integrates all deduplication methods including Claude"""
    
    def __init__(self):
        self.embedding_matcher = AIEmbeddingMatcher() if AI_EMBEDDINGS_AVAILABLE else None
        self.ml_predictor = MLDuplicatePredictor() if ML_AVAILABLE else None
        self.llm_validator = ClaudeLLMValidator() if LLM_AVAILABLE else None  # Now using Claude
        
        # Adjusted weights with Claude's superior understanding
        self.weights = {
            'fuzzy': 0.25,
            'semantic': 0.25,
            'ml': 0.20,
            'llm': 0.30  # Higher weight for Claude's nuanced understanding
        }
    
    def find_all_duplicates(self, records: List[dict]) -> List[dict]:
        """Find duplicates using all available methods including Claude validation"""
        all_matches = {}
        
        logger.info("🔍 Starting AI-enhanced deduplication with Claude...")
        
        # [Previous matching logic remains the same...]
        
        # Claude validation for uncertain cases
        if self.llm_validator:
            uncertain_matches = [
                m for m in all_matches.values() 
                if 0.5 < m.get('combined_score', 0) < 0.85
            ]
            
            if uncertain_matches:
                logger.info(f"🤖 Sending {len(uncertain_matches)} uncertain matches to Claude...")
                
                for match in uncertain_matches:
                    llm_result = self.llm_validator.validate_match(
                        match['record1'], 
                        match['record2'], 
                        match['combined_score']
                    )
                    
                    if llm_result.get('use_llm'):
                        # Update scores with Claude's assessment
                        match['scores']['llm'] = llm_result['confidence']
                        match['llm_reasoning'] = llm_result.get('reasoning', '')
                        match['relationship_type'] = llm_result.get('relationship', 'unknown')
                        
                        # Recalculate combined score with Claude's input
                        match['combined_score'] = self._calculate_combined_score(match['scores'])
                        
                        # Claude can override if very confident
                        if llm_result['confidence'] > 0.9:
                            match['is_duplicate'] = llm_result['is_duplicate']
                            match['override_reason'] = 'claude_high_confidence'
        
        # [Rest of the method remains the same...]
        
        return final_matches

# ============================================
# USAGE STATISTICS TRACKER
# ============================================

class DeduplicationStats:
    """Track and report deduplication statistics"""
    
    @staticmethod
    def generate_report(matches: List[dict], records_count: int) -> dict:
        """Generate comprehensive statistics report"""
        
        duplicates = [m for m in matches if m.get('is_duplicate')]
        claude_validated = [m for m in matches if m.get('llm_validation')]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': records_count,
            'duplicate_pairs_found': len(duplicates),
            'ai_features_used': {
                'fuzzy_matching': True,
                'semantic_embeddings': any('semantic' in m.get('scores', {}) for m in matches),
                'ml_model': any('ml' in m.get('scores', {}) for m in matches),
                'claude_validation': len(claude_validated) > 0
            },
            'claude_statistics': {
                'validations_performed': len(claude_validated),
                'high_confidence_overrides': sum(1 for m in claude_validated if m.get('override_reason') == 'claude_high_confidence'),
                'relationship_types': Counter([m.get('relationship_type', 'unknown') for m in claude_validated])
            },
            'confidence_distribution': {
                'high_90_100': sum(1 for m in duplicates if m.get('combined_score', 0) >= 0.9),
                'medium_75_90': sum(1 for m in duplicates if 0.75 <= m.get('combined_score', 0) < 0.9),
                'low_50_75': sum(1 for m in duplicates if 0.5 <= m.get('combined_score', 0) < 0.75)
            }
        }
        
        return report
