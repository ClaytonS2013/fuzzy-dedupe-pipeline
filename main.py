"""
Fuzzy Matching Deduplication Pipeline - Main Entry Point
FIXED: Proper client initialization and function calls
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime
from supabase import create_client, Client
import gspread
from google.oauth2.service_account import Credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import sheets_sync module
try:
    from sheets_sync.sync import sync_sheets_to_supabase, sync_supabase_to_sheets
except ImportError as e:
    logging.error(f"❌ Failed to import sheets_sync module: {e}")
    sync_sheets_to_supabase = None
    sync_supabase_to_sheets = None

# Environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID')
GOOGLE_CREDENTIALS = os.environ.get('GOOGLE_CREDENTIALS')

def init_google_sheets_client():
    """Initialize Google Sheets client with credentials"""
    try:
        # Parse credentials from environment variable
        if GOOGLE_CREDENTIALS.startswith('{'):
            # It's JSON string
            import json
            creds_dict = json.loads(GOOGLE_CREDENTIALS)
        else:
            # It might be base64 encoded
            import base64
            creds_json = base64.b64decode(GOOGLE_CREDENTIALS).decode('utf-8')
            creds_dict = json.loads(creds_json)
        
        # Create credentials object
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets', 
                   'https://www.googleapis.com/auth/drive']
        )
        
        # Authorize and return client
        client = gspread.authorize(creds)
        logging.info("✅ Google Sheets client initialized successfully")
        return client
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize Google Sheets client: {e}")
        raise

def init_supabase_client():
    """Initialize Supabase client"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✅ Supabase client initialized successfully")
        return supabase
    except Exception as e:
        logging.error(f"❌ Failed to initialize Supabase client: {e}")
        raise

def log_pipeline_stage(stage_name, status, records_processed=0, error_message=None, duration_ms=None):
    """Log pipeline execution to Supabase dedupe_log table"""
    try:
        log_data = {
            "stage_name": stage_name,
            "status": status,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat() if status in ["success", "failed"] else None,
            "records_processed": records_processed,
            "error_message": error_message,
            "duration_ms": duration_ms
        }
        
        # Remove None values to avoid issues
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        # Debug output
        logging.info(f"🔍 DEBUG - Logging to dedupe_log: {json.dumps(log_data, indent=2)}")
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/dedupe_log",
            headers=headers,
            json=log_data
        )
        
        if response.status_code >= 400:
            logging.warning(f"⚠️ Failed to log stage {stage_name} to dedupe_log: {response.status_code} {response.reason}")
            logging.warning(f"🔍 DEBUG - Response content: {response.text}")
        else:
            logging.info(f"✅ Successfully logged stage {stage_name} to dedupe_log")
        
    except Exception as e:
        logging.warning(f"⚠️ Failed to log stage {stage_name} to dedupe_log: {str(e)}")

def run_pipeline():
    """Run the complete data pipeline"""
    start_time = time.time()
    
    logging.info("🚀 Fuzzy Matching Antelligence Pipeline Starting")
    logging.info("============================================================")
    
    # Verify environment variables
    missing_vars = []
    if not SUPABASE_URL:
        missing_vars.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing_vars.append("SUPABASE_KEY")
    if not SPREADSHEET_ID:
        missing_vars.append("SPREADSHEET_ID")
    if not GOOGLE_CREDENTIALS:
        missing_vars.append("GOOGLE_CREDENTIALS")
    
    if missing_vars:
        logging.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logging.error("Pipeline cannot proceed without these variables")
        return
    
    logging.info("✅ Environment variables verified")
    
    # Initialize clients
    try:
        sheets_client = init_google_sheets_client()
        supabase_client = init_supabase_client()
    except Exception as e:
        logging.error(f"❌ Failed to initialize clients: {e}")
        return
    
    logging.info("")
    
    # Stage 1: Google Sheets → Supabase
    logging.info("============================================================")
    logging.info("📥 STAGE 1: Google Sheets → Supabase Sync")
    logging.info("============================================================")
    
    stage1_start = time.time()
    if sync_sheets_to_supabase:
        try:
            # Pass the initialized clients and spreadsheet ID
            records_processed = sync_sheets_to_supabase(sheets_client, supabase_client)
            stage1_duration = int((time.time() - stage1_start) * 1000)
            logging.info("============================================================")
            logging.info(f"✅ Stage 1 Complete: Synced {records_processed} records (took {stage1_duration} ms)")
            logging.info("============================================================")
            log_pipeline_stage("sheets_to_supabase", "success", records_processed, duration_ms=stage1_duration)
        except Exception as e:
            stage1_duration = int((time.time() - stage1_start) * 1000)
            error_msg = str(e)
            logging.error("============================================================")
            logging.error(f"❌ Stage 1 Failed: {error_msg}")
            logging.error("============================================================")
            logging.error("Pipeline will continue but data may be stale")
            log_pipeline_stage("sheets_to_supabase", "failed", 0, error_msg, stage1_duration)
    else:
        logging.error("============================================================")
        logging.error("❌ Stage 1 Failed: sheets_sync module not available")
        logging.error("============================================================")
    
    # Stage 2: Deduplication
    logging.info("")
    logging.info("============================================================")
    logging.info("🧹 STAGE 2: Dedupe Processing")
    logging.info("============================================================")
    
    stage2_start = time.time()
    try:
        # Check if dedupe module is available
        try:
            from dedupe_logic.processor import run_deduplication
            records_processed = run_deduplication(supabase_client)
            stage2_duration = int((time.time() - stage2_start) * 1000)
            logging.info("============================================================")
            logging.info(f"✅ Stage 2 Complete: Processed {records_processed} records (took {stage2_duration} ms)")
            logging.info("============================================================")
            log_pipeline_stage("dedupe", "success", records_processed, duration_ms=stage2_duration)
        except ImportError:
            logging.warning("⚠️ Skipping Stage 2: Dedupe not available")
            log_pipeline_stage("dedupe", "skipped", 0, "Dedupe module not available")
    except Exception as e:
        stage2_duration = int((time.time() - stage2_start) * 1000)
        error_msg = str(e)
        logging.error(f"❌ Stage 2 Failed: {error_msg}")
        log_pipeline_stage("dedupe", "failed", 0, error_msg, stage2_duration)
    
    # Stage 3: Supabase → Google Sheets
    logging.info("")
    logging.info("============================================================")
    logging.info("📤 STAGE 3: Supabase → Google Sheets Writeback")
    logging.info("============================================================")
    
    stage3_start = time.time()
    if sync_supabase_to_sheets:
        try:
            # Pass the initialized clients
            records_processed = sync_supabase_to_sheets(sheets_client, supabase_client)
            stage3_duration = int((time.time() - stage3_start) * 1000)
            logging.info("============================================================")
            logging.info(f"✅ Stage 3 Complete: Wrote {records_processed} rows to Google Sheets (took {stage3_duration} ms)")
            logging.info("============================================================")
            log_pipeline_stage("supabase_to_sheets", "success", records_processed, duration_ms=stage3_duration)
        except Exception as e:
            stage3_duration = int((time.time() - stage3_start) * 1000)
            error_msg = str(e)
            logging.error(f"❌ Stage 3 Failed: {error_msg}")
            log_pipeline_stage("supabase_to_sheets", "failed", 0, error_msg, stage3_duration)
    else:
        logging.error("============================================================")
        logging.error("❌ Stage 3 Failed: sheets_sync module not available")
        logging.error("============================================================")
    
    # Pipeline complete
    total_duration = int((time.time() - start_time) * 1000)
    logging.info("")
    logging.info("============================================================")
    logging.info("🏁 Pipeline Finished")
    logging.info("============================================================")
    
    # Report any skipped stages
    skipped_stages = []
    if sync_sheets_to_supabase is None:
        skipped_stages.append("Google Sheets → Supabase sync unavailable")
    if sync_supabase_to_sheets is None:
        skipped_stages.append("Supabase → Google Sheets sync unavailable")
    
    if skipped_stages:
        logging.warning("⚠️ Some stages were skipped or failed")
        for stage in skipped_stages:
            logging.warning(f"  - {stage}")

if __name__ == "__main__":
    # Print startup message
    print("🚀 Starting Fuzzy Matching Pipeline...")
    print("==================================")
    run_pipeline()
