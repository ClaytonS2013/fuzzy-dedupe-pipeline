"""
Fuzzy Matching Pipeline - Main Entry Point
Connects Google Sheets to Supabase with deduplication
"""

import os
import sys
import json
import base64
import logging
from datetime import datetime
import traceback
from google.oauth2 import service_account
import gspread
from supabase import create_client, Client
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment_variables() -> bool:
    """Verify all required environment variables are set"""
    required_vars = ['GOOGLE_CREDENTIALS', 'SPREADSHEET_ID', 'SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            logger.error(f"Missing environment variable: {var}")
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("Environment variables verified")
    logger.info(f"   SUPABASE_URL: {os.getenv('SUPABASE_URL')[:40]}...")
    logger.info(f"   SPREADSHEET_ID: {os.getenv('SPREADSHEET_ID')}")
    logger.info(f"   GOOGLE_CREDENTIALS length: {len(os.getenv('GOOGLE_CREDENTIALS', ''))}")
    
    return True

def init_google_sheets_client():
    """Initialize Google Sheets client with service account credentials"""
    try:
        logger.info("GOOGLE_CREDENTIALS length: %d", len(os.getenv('GOOGLE_CREDENTIALS', '')))
        
        # Decode base64 credentials
        logger.info("Decoding base64 credentials...")
        creds_base64 = os.getenv('GOOGLE_CREDENTIALS')
        creds_json = base64.b64decode(creds_base64).decode('utf-8')
        logger.info("Successfully decoded base64 credentials")
        
        # Parse JSON
        service_account_info = json.loads(creds_json)
        
        # Create credentials
        logger.info("Creating credentials object...")
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        # Authorize client
        logger.info("Authorizing with Google...")
        client = gspread.authorize(credentials)
        logger.info("Google Sheets client initialized successfully")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Google Sheets client: {str(e)}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise

def init_supabase_client() -> Client:
    """Initialize Supabase client"""
    try:
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        client = create_client(url, key)
        logger.info("Supabase client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        raise

def run_pipeline():
    """Main pipeline orchestration"""
    start_time = datetime.now()
    
    # Check environment
    if not check_environment_variables():
        logger.error("Environment check failed")
        return
    
    logger.info("\nInitializing clients...")
    
    # Initialize clients
    try:
        sheets_client = init_google_sheets_client()
        supabase_client = init_supabase_client()
        logger.info("All clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {str(e)}")
        return
    
    # Import pipeline stages
    from sheets_sync.sync import sync_sheets_to_supabase, sync_supabase_to_sheets
    from dedupe_logic.processor import run_deduplication
    
    # Log to dedupe_log table
    def log_stage(stage_name: str, status: str, start: datetime, end: datetime = None, 
                  records: int = 0, error: str = None):
        try:
            end = end or datetime.now()
            duration_ms = int((end - start).total_seconds() * 1000)
            
            log_entry = {
                "stage_name": stage_name,
                "status": status,
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "records_processed": records,
                "duration_ms": duration_ms
            }
            
            if error:
                log_entry["error_message"] = error
            
            logger.info(f"DEBUG - Logging to dedupe_log: {json.dumps(log_entry, indent=2)}")
            
            response = supabase_client.table('dedupe_log').insert(log_entry).execute()
            logger.info(f"Successfully logged stage {stage_name} to dedupe_log")
        except Exception as e:
            logger.error(f"Failed to log stage {stage_name}: {str(e)}")
    
    # Stage 1: Google Sheets to Supabase
    logger.info("=" * 60)
    logger.info("STAGE 1: Google Sheets to Supabase Sync")
    logger.info("=" * 60)
    
    stage_start = datetime.now()
    try:
        records_processed = sync_sheets_to_supabase(sheets_client, supabase_client)
        stage_end = datetime.now()
        duration = int((stage_end - stage_start).total_seconds() * 1000)
        logger.info("=" * 60)
        logger.info(f"Stage 1 Complete: {records_processed} records synced (took {duration} ms)")
        logger.info("=" * 60)
        log_stage("sheets_to_supabase", "success", stage_start, stage_end, records_processed)
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Stage 1 Failed: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        logger.error("=" * 60)
        logger.error("Pipeline will continue but data may be stale")
        log_stage("sheets_to_supabase", "failed", stage_start, error=str(e))
    
    # Stage 2: Dedupe Processing
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Dedupe Processing")
    logger.info("=" * 60)
    
    stage_start = datetime.now()
    try:
        dedupe_count = run_deduplication(supabase_client)
        stage_end = datetime.now()
        duration = int((stage_end - stage_start).total_seconds() * 1000)
        logger.info("=" * 60)
        logger.info(f"Stage 2 Complete: Processed {dedupe_count} records (took {duration} ms)")
        logger.info("=" * 60)
        log_stage("dedupe", "success", stage_start, stage_end, dedupe_count)
    except Exception as e:
        logger.error(f"Stage 2 Failed: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        log_stage("dedupe", "failed", stage_start, error=str(e))
    
    # Stage 3: Supabase to Google Sheets
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Supabase to Google Sheets Writeback")
    logger.info("=" * 60)
    
    stage_start = datetime.now()
    try:
        records_written = sync_supabase_to_sheets(supabase_client, sheets_client)
        stage_end = datetime.now()
        duration = int((stage_end - stage_start).total_seconds() * 1000)
        logger.info("=" * 60)
        logger.info(f"Stage 3 Complete: Wrote {records_written} rows to Google Sheets (took {duration} ms)")
        logger.info("=" * 60)
        log_stage("supabase_to_sheets", "success", stage_start, stage_end, records_written)
    except Exception as e:
        logger.error(f"Stage 3 Failed: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        log_stage("supabase_to_sheets", "failed", stage_start, error=str(e))
    
    # Final summary
    total_time = int((datetime.now() - start_time).total_seconds() * 1000)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Pipeline Finished - Total time: {total_time}ms")
    logger.info("=" * 60)

if __name__ == "__main__":
    logger.info("Fuzzy Matching Antelligence Pipeline Starting")
    logger.info("=" * 60)
    
    try:
        run_pipeline()
        logger.info("Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
