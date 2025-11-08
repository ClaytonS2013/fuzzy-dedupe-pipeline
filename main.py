import os
import sys
import json
import time
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import sheets_sync module
try:
    from sheets_sync.main import sync_sheets_to_supabase, sync_supabase_to_sheets
except ImportError:
    logging.error("❌ Failed to import sheets_sync module")
    sync_sheets_to_supabase = None
    sync_supabase_to_sheets = None

# Environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID')

def log_pipeline_stage(stage_name, status, records_processed=0, error_message=None, duration_ms=None):
    """Log pipeline execution to Supabase dedupe_log table"""
    try:
        log_data = {
            "stage_name": stage_name,
            "status": status,
            "records_processed": records_processed,
            "error_message": error_message,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }
        
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
            logging.warning(f"⚠️ Failed to log stage {stage_name} to dedupe_log: {response.status_code} {response.reason} for url: {response.url}")
            logging.warning(f"🔍 DEBUG - Response content: {response.text}")
        
    except Exception as e:
        logging.warning(f"⚠️ Failed to log stage {stage_name} to dedupe_log: {str(e)}")

def run_pipeline():
    """Run the complete data pipeline"""
    start_time = time.time()
    
    logging.info("🚀 Fuzzy Matching Antelligence Pipeline Starting")
    logging.info("============================================================")
    
    # Verify environment variables
    if not all([SUPABASE_URL, SUPABASE_KEY, SPREADSHEET_ID]):
        logging.error("❌ Missing required environment variables")
        return
    logging.info("✅ Environment variables verified")
    logging.info("")
    
    # Stage 1: Google Sheets → Supabase
    logging.info("============================================================")
    logging.info("📥 STAGE 1: Google Sheets → Supabase Sync")
    logging.info("============================================================")
    
    stage1_start = time.time()
    if sync_sheets_to_supabase:
        try:
            records_processed = sync_sheets_to_supabase(
                SUPABASE_URL, 
                SUPABASE_KEY,
                SPREADSHEET_ID
            )
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
        logging.error("❌ Stage 1 Failed: sheets_sync module not available")
    
    # Stage 2: Deduplication
    logging.info("")
    logging.info("============================================================")
    logging.info("🧹 STAGE 2: Dedupe Processing")
    logging.info("============================================================")
    
    stage2_start = time.time()
    try:
        # Check if dedupe module is available
        try:
            # This would normally import the dedupe module
            # from dedupe.main import run_deduplication
            raise ImportError("Dedupe module not implemented yet")
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
            records_processed = sync_supabase_to_sheets(
                SUPABASE_URL, 
                SUPABASE_KEY,
                SPREADSHEET_ID
            )
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
        logging.error("❌ Stage 3 Failed: sheets_sync module not available")
    
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
    
    # Always warn that dedupe is skipped since we don't have that module yet
    skipped_stages.append("Dedupe unavailable")
    
    if skipped_stages:
        logging.warning("⚠️ Some stages were skipped")
        for stage in skipped_stages:
            logging.warning(f"  - {stage}")

if __name__ == "__main__":
    run_pipeline()
