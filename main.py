"""
Fuzzy Matching Deduplication Pipeline - Main Entry Point
FINAL VERSION: All imports fixed, error handling improved
"""

import os
import sys
import json
import base64
import time
import logging
import traceback
import requests
from datetime import datetime

# Import these before using them
try:
    from supabase import create_client, Client
except ImportError as e:
    print(f"ERROR: Failed to import supabase: {e}")
    print("Installing supabase...")
    os.system("pip install supabase")
    from supabase import create_client, Client

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError as e:
    print(f"ERROR: Failed to import Google libraries: {e}")
    print("Installing required packages...")
    os.system("pip install gspread google-auth google-auth-oauthlib")
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
        if not GOOGLE_CREDENTIALS:
            raise ValueError("GOOGLE_CREDENTIALS environment variable is not set")
            
        # Debug log
        logging.info(f"📝 GOOGLE_CREDENTIALS length: {len(GOOGLE_CREDENTIALS)}")
        
        # Parse credentials from environment variable
        creds_dict = None
        if GOOGLE_CREDENTIALS.startswith('{'):
            # It's JSON string
            logging.info("📄 Parsing as JSON string...")
            creds_dict = json.loads(GOOGLE_CREDENTIALS)
        else:
            # It's base64 encoded
            logging.info("🔐 Decoding base64 credentials...")
            try:
                creds_json = base64.b64decode(GOOGLE_CREDENTIALS).decode('utf-8')
                creds_dict = json.loads(creds_json)
                logging.info("✅ Successfully decoded base64 credentials")
            except Exception as decode_error:
                logging.error(f"❌ Failed to decode base64: {decode_error}")
                raise
        
        if not creds_dict:
            raise ValueError("Failed to parse credentials")
            
        # Create credentials object
        logging.info("🔑 Creating credentials object...")
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets', 
                   'https://www.googleapis.com/auth/drive']
        )
        
        # Authorize and return client
        logging.info("🔗 Authorizing with Google...")
        client = gspread.authorize(creds)
        logging.info("✅ Google Sheets client initialized successfully")
        return client
        
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON decode error: {e}")
        logging.error(f"   First 100 chars: {GOOGLE_CREDENTIALS[:100]}...")
        raise
    except Exception as e:
        logging.error(f"❌ Failed to initialize Google Sheets client: {e}")
        logging.error(f"   Error type: {type(e).__name__}")
        logging.error(f"   Traceback: {traceback.format_exc()}")
        raise

def init_supabase_client():
    """Initialize Supabase client"""
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(f"Missing Supabase credentials. URL: {SUPABASE_URL}, KEY: {'set' if SUPABASE_KEY else 'not set'}")
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("✅ Supabase client initialized successfully")
        return supabase
    except Exception as e:
        logging.error(f"❌ Failed to initialize Supabase client: {e}")
        raise

def log_pipeline_stage(stage_name, status, records_processed=0, error_message=None, duration_ms=None):
    """Log pipeline execution to Supabase dedupe_log table"""
    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            logging.warning("⚠️ Cannot log to Supabase: credentials not available")
            return
            
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
    logging.info(f"   SUPABASE_URL: {SUPABASE_URL[:30]}...")
    logging.info(f"   SPREADSHEET_ID: {SPREADSHEET_ID}")
    logging.info(f"   GOOGLE_CREDENTIALS length: {len(GOOGLE_CREDENTIALS)}")
    
    # Initialize clients
    sheets_client = None
    supabase_client = None
    
    try:
        logging.info("\n🔧 Initializing clients...")
        sheets_client = init_google_sheets_client()
        supabase_client = init_supabase_client()
        logging.info("✅ All clients initialized successfully\n")
    except Exception as e:
        logging.error(f"❌ Failed to initialize clients: {e}")
        logging.error(f"   Traceback: {traceback.format_exc()}")
        return
    
    # Stage 1: Google Sheets → Supabase
    logging.info("============================================================")
    logging.info("📥 STAGE 1: Google Sheets → Supabase Sync")
    logging.info("============================================================")
    
    stage1_start = time.time()
    if sync_sheets_to_supabase and sheets_client and supabase_client:
        try:
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
            logging.error(f"   Traceback: {traceback.format_exc()}")
            logging.error("============================================================")
            logging.error("Pipeline will continue but data may be stale")
            log_pipeline_stage("sheets_to_supabase", "failed", 0, error_msg, stage1_duration)
    else:
        logging.error("============================================================")
        logging.error("❌ Stage 1 Failed: sheets_sync module or clients not available")
        logging.error("============================================================")
    
    # Stage 2: Deduplication
    logging.info("")
    logging.info("============================================================")
    logging.info("🧹 STAGE 2: Dedupe Processing")
    logging.info("============================================================")
    
    stage2_start = time.time()
    if supabase_client:
        try:
            from dedupe_logic.processor import run_deduplication
            records_processed = run_deduplication(supabase_client)
            stage2_duration = int((time.time() - stage2_start) * 1000)
            logging.info("============================================================")
            logging.info(f"✅ Stage 2 Complete: Processed {records_processed} records (took {stage2_duration} ms)")
            logging.info("============================================================")
            log_pipeline_stage("dedupe", "success", records_processed, duration_ms=stage2_duration)
        except ImportError as e:
            logging.warning(f"⚠️ Skipping Stage 2: Dedupe module not available - {e}")
            log_pipeline_stage("dedupe", "skipped", 0, f"Dedupe module not available: {e}")
        except Exception as e:
            stage2_duration = int((time.time() - stage2_start) * 1000)
            error_msg = str(e)
            logging.error(f"❌ Stage 2 Failed: {error_msg}")
            logging.error(f"   Traceback: {traceback.format_exc()}")
            log_pipeline_stage("dedupe", "failed", 0, error_msg, stage2_duration)
    else:
        logging.error("❌ Stage 2 Skipped: Supabase client not available")
    
    # Stage 3: Supabase → Google Sheets
    logging.info("")
    logging.info("============================================================")
    logging.info("📤 STAGE 3: Supabase → Google Sheets Writeback")
    logging.info("============================================================")
    
    stage3_start = time.time()
    if sync_supabase_to_sheets and sheets_client and supabase_client:
        try:
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
            logging.error(f"   Traceback: {traceback.format_exc()}")
            log_pipeline_stage("supabase_to_sheets", "failed", 0, error_msg, stage3_duration)
    else:
        logging.error("============================================================")
        logging.error("❌ Stage 3 Failed: sheets_sync module or clients not available")
        logging.error("============================================================")
    
    # Pipeline complete
    total_duration = int((time.time() - start_time) * 1000)
    logging.info("")
    logging.info("============================================================")
    logging.info(f"🏁 Pipeline Finished - Total time: {total_duration}ms")
    logging.info("============================================================")

if __name__ == "__main__":
    # Print startup message
    print("🚀 Starting Fuzzy Matching Pipeline...")
    print("==================================")
    try:
        run_pipeline()
    except Exception as e:
        print(f"❌ Pipeline crashed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        sys.exit(1)
