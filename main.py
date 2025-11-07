"""
main.py

Fuzzy Matching Antelligence Pipeline
Orchestrates: Google Sheets → Supabase → Deduplication → Google Sheets
"""

import os
import sys
import logging
from pathlib import Path
import time
import requests
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline logging helper
#
# This function writes a single stage entry to the `dedupe_log` table in
# Supabase. Each log entry records the stage name, number of rows processed,
# duration in milliseconds, status ("success" or "error"), and an optional
# error message. The log table is assumed to exist in Supabase. If any
# errors occur while sending the log, they are printed to the local logs but
# do not interrupt pipeline execution.
def _log_pipeline_stage(
    stage: str,
    row_count: int,
    duration_ms: int,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """Send a pipeline stage log to Supabase."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        logger.warning("⚠️ Cannot log to Supabase: SUPABASE_URL or SUPABASE_KEY not set")
        return
    # Prepare log record
    import datetime
    log_record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "row_count": row_count,
        "duration_ms": duration_ms,
        "status": status,
        "error_message": error_message or "",
    }
    url = f"{supabase_url}/rest/v1/dedupe_log"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    try:
        resp = requests.post(url, json=[log_record], headers=headers, timeout=10)
        resp.raise_for_status()
        logger.debug(f"Logged stage {stage} to dedupe_log")
    except Exception as exc:
        logger.warning(f"⚠️ Failed to log stage {stage} to dedupe_log: {exc}")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import sync functions
try:
    from sheets_sync.main import (
        sync_sheets_to_supabase,
        write_clean_data_to_sheet
    )
    SHEETS_SYNC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Sheets sync module not available: {e}")
    SHEETS_SYNC_AVAILABLE = False

# Import dedupe function (adjust this import based on your actual dedupe file)
try:
    # Try common variations of dedupe module names
    dedupe_module = None
    for module_name in ['dedupe_pipeline', 'dedupe', 'main_dedupe', 'fuzzy_dedupe']:
        try:
            if module_name == 'dedupe_pipeline':
                from dedupe_pipeline import run_dedupe
                dedupe_module = 'dedupe_pipeline'
            elif module_name == 'dedupe':
                from dedupe import run_dedupe
                dedupe_module = 'dedupe'
            elif module_name == 'main_dedupe':
                from main_dedupe import run_dedupe
                dedupe_module = 'main_dedupe'
            elif module_name == 'fuzzy_dedupe':
                from fuzzy_dedupe import run_dedupe
                dedupe_module = 'fuzzy_dedupe'
            
            logger.info(f"✅ Loaded dedupe module: {dedupe_module}")
            DEDUPE_AVAILABLE = True
            break
        except ImportError:
            continue
    
    if dedupe_module is None:
        raise ImportError("Could not find dedupe module")
        
except ImportError as e:
    logger.warning(f"⚠️ Dedupe module not available: {e}")
    DEDUPE_AVAILABLE = False
    
    # Define a dummy function
    def run_dedupe():
        logger.warning("⚠️ Dedupe function not available - skipping deduplication")


def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("🚀 Fuzzy Matching Antelligence Pipeline Starting")
    logger.info("=" * 60)
    
    # Verify environment variables
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SPREADSHEET_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Pipeline cannot proceed without these variables")
        sys.exit(1)
    
    logger.info("✅ Environment variables verified")
    
    # Stage 1: Google Sheets → Supabase
    if SHEETS_SYNC_AVAILABLE:
        try:
            start_time = time.perf_counter()
            logger.info("")
            logger.info("=" * 60)
            logger.info("📥 STAGE 1: Google Sheets → Supabase Sync")
            logger.info("=" * 60)

            row_count = sync_sheets_to_supabase()

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info("=" * 60)
            logger.info(f"✅ Stage 1 Complete: Synced {row_count} rows to Supabase (took {duration_ms} ms)")
            logger.info("=" * 60)

            # Log to Supabase
            _log_pipeline_stage(
                stage="sheets_to_supabase",
                row_count=row_count,
                duration_ms=duration_ms,
                status="success",
                error_message=None
            )
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Stage 1 Failed: {e}")
            logger.error("=" * 60)
            logger.error("Pipeline will continue but data may be stale")

            # Log failure to Supabase
            _log_pipeline_stage(
                stage="sheets_to_supabase",
                row_count=0,
                duration_ms=0,
                status="error",
                error_message=str(e)
            )
    else:
        logger.warning("⚠️ Skipping Stage 1: Sheets sync not available")
    
    # Stage 2: Run Deduplication
    if DEDUPE_AVAILABLE:
        try:
            start_time = time.perf_counter()
            logger.info("")
            logger.info("=" * 60)
            logger.info("🧐 STAGE 2: Running Deduplication")
            logger.info("=" * 60)
            
            run_dedupe()
            
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info("=" * 60)
            logger.info(f"✅ Stage 2 Complete: Deduplication finished (took {duration_ms} ms)")
            logger.info("=" * 60)

            _log_pipeline_stage(
                stage="deduplication",
                row_count=0,
                duration_ms=duration_ms,
                status="success",
                error_message=None
            )
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Stage 2 Failed: {e}")
            logger.error("=" * 60)
            logger.error("Pipeline will continue to writeback stage")

            # Log failure to Supabase
            _log_pipeline_stage(
                stage="deduplication",
                row_count=0,
                duration_ms=0,
                status="error",
                error_message=str(e)
            )
    else:
        logger.warning("⚠️ Skipping Stage 2: Dedupe not available")
    
    # Stage 3: Supabase → Google Sheets (Clean Data)
    if SHEETS_SYNC_AVAILABLE:
        try:
            start_time = time.perf_counter()
            logger.info("")
            logger.info("=" * 60)
            logger.info("📤 STAGE 3: Supabase → Google Sheets Writeback")
            logger.info("=" * 60)

            row_count = write_clean_data_to_sheet()

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info("=" * 60)
            logger.info(f"✅ Stage 3 Complete: Wrote {row_count} rows to Google Sheets (took {duration_ms} ms)")
            logger.info("=" * 60)

            _log_pipeline_stage(
                stage="supabase_to_sheets",
                row_count=row_count,
                duration_ms=duration_ms,
                status="success",
                error_message=None
            )
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Stage 3 Failed: {e}")
            logger.error("=" * 60)

            # Log failure to Supabase
            _log_pipeline_stage(
                stage="supabase_to_sheets",
                row_count=0,
                duration_ms=0,
                status="error",
                error_message=str(e)
            )
    else:
        logger.warning("⚠️ Skipping Stage 3: Sheets sync not available")
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("🏁 Pipeline Finished")
    logger.info("=" * 60)
    
    if SHEETS_SYNC_AVAILABLE and DEDUPE_AVAILABLE:
        logger.info("✅ All stages completed")
    else:
        logger.warning("⚠️ Some stages were skipped")
        if not SHEETS_SYNC_AVAILABLE:
            logger.warning("  - Sheets sync unavailable")
        if not DEDUPE_AVAILABLE:
            logger.warning("  - Dedupe unavailable")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)
