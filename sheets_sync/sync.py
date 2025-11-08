"""
Sheets Sync Module - Core Synchronization Functions
Handles bidirectional sync between Google Sheets and Supabase
"""

import os
import json
import base64
import logging
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from supabase import create_client

logger = logging.getLogger(__name__)


def init_google_sheets_client():
    """Initialize Google Sheets client with service account credentials"""
    try:
        # Get credentials from environment
        creds_b64 = os.getenv('GOOGLE_CREDENTIALS')
        
        if not creds_b64:
            raise ValueError("GOOGLE_CREDENTIALS environment variable not set")
        
        # Check if credentials are base64 encoded
        try:
            # Try to decode as base64
            logger.info("🔑 Creating service account file from base64 credentials...")
            creds_json = base64.b64decode(creds_b64).decode('utf-8')
            logger.info("✅ Service account file created")
        except Exception:
            # If not base64, assume it's already JSON string
            logger.info("🔑 Using credentials as JSON string...")
            creds_json = creds_b64
        
        # Parse credentials
        creds_dict = json.loads(creds_json)
        
        # Create credentials object
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        
        # Initialize gspread client
        client = gspread.authorize(credentials)
        
        logger.info("✅ Google Sheets client initialized successfully")
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Sheets client: {e}")
        raise


def init_supabase_client():
    """Initialize Supabase client"""
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY not set")
        
        client = create_client(supabase_url, supabase_key)
        logger.info("✅ Supabase client initialized successfully")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {e}")
        raise


def sync_sheets_to_supabase(sheets_client, supabase, batch_size=100):
    """
    Sync data from Google Sheets to Supabase
    
    Args:
        sheets_client: Initialized gspread client
        supabase: Initialized Supabase client
        batch_size: Number of records to insert per batch
        
    Returns:
        int: Number of records synced
    """
    logger.info("📥 Starting Google Sheets → Supabase sync...")
    
    try:
        # Open spreadsheet
        spreadsheet_id = os.getenv('SPREADSHEET_ID')
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        
        # Get the "Raw_Practices" worksheet
        worksheet = spreadsheet.worksheet('Raw_Practices')
        
        # Get all records
        records = worksheet.get_all_records()
        logger.info(f"📊 Found {len(records)} rows in Raw_Practices")
        
        if not records:
            logger.warning("⚠️ No records found in Google Sheets")
            return 0
        
        # Clear existing records in Supabase
        logger.info("🗑️ Clearing existing practice_records...")
        try:
            supabase.table('practice_records').delete().neq('id', 0).execute()
        except Exception as e:
            logger.warning(f"⚠️ Could not clear existing records: {e}")
        
        # Convert records to proper format
        # Handle potential column name variations
        processed_records = []
        for record in records:
            # Create a clean record dict, handling common column name patterns
            clean_record = {}
            for key, value in record.items():
                # Convert column names to lowercase with underscores
                clean_key = key.lower().replace(' ', '_').replace('-', '_')
                
                # Skip empty values
                if value == '':
                    clean_record[clean_key] = None
                else:
                    clean_record[clean_key] = value
            
            processed_records.append(clean_record)
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(processed_records), batch_size):
            batch = processed_records[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                response = supabase.table('practice_records').insert(batch).execute()
                inserted_count = len(batch)
                total_inserted += inserted_count
                logger.info(f"✅ Inserted batch {batch_num}: {inserted_count} records")
                
            except Exception as e:
                logger.error(f"❌ Failed to insert batch {batch_num}: {e}")
                # Log the structure of the first record for debugging
                if batch:
                    logger.error(f"🔍 Sample record structure: {list(batch[0].keys())}")
                raise
        
        logger.info(f"✅ Sheets → Supabase sync completed: {total_inserted} records")
        return total_inserted
        
    except Exception as e:
        logger.error(f"❌ Sheets → Supabase sync failed: {e}")
        raise


def sync_supabase_to_sheets(sheets_client, supabase):
    """
    Sync deduplicated data from Supabase back to Google Sheets
    
    Args:
        sheets_client: Initialized gspread client
        supabase: Initialized Supabase client
        
    Returns:
        int: Number of rows written
    """
    logger.info("📤 Starting Supabase → Google Sheets writeback...")
    
    try:
        # Fetch dedupe results from Supabase
        logger.info("🔍 Fetching dedupe_results from Supabase...")
        response = supabase.table('dedupe_results').select('*').execute()
        results = response.data
        
        logger.info(f"📊 Found {len(results)} dedupe results")
        
        if not results:
            logger.warning("⚠️ No dedupe results found")
            return 0
        
        # Open spreadsheet
        spreadsheet_id = os.getenv('SPREADSHEET_ID')
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        
        # Get or create "Clean Data" worksheet
        try:
            worksheet = spreadsheet.worksheet('Clean Data')
        except gspread.exceptions.WorksheetNotFound:
            logger.info("📝 Creating 'Clean Data' worksheet...")
            worksheet = spreadsheet.add_worksheet(title='Clean Data', rows=1000, cols=26)
        
        # Clear existing data
        logger.info("🗑️ Clearing Clean Data worksheet...")
        worksheet.clear()
        
        # Prepare data for writing
        if results:
            # Get column headers from first record
            headers = list(results[0].keys())
            
            # Convert results to list of lists
            rows = [headers]  # Header row
            for result in results:
                row = [result.get(header, '') for header in headers]
                rows.append(row)
            
            # Write to sheet
            logger.info(f"✍️ Writing {len(rows)} rows to Clean Data worksheet...")
            worksheet.update('A1', rows)
            
            logger.info(f"✅ Supabase → Google Sheets writeback completed: {len(results)} records")
            return len(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Supabase → Google Sheets writeback failed: {e}")
        raise


# Utility functions

def get_sheet_columns(sheets_client, worksheet_name='Raw_Practices'):
    """Get column names from a Google Sheet"""
    try:
        spreadsheet_id = os.getenv('SPREADSHEET_ID')
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        
        # Get first row (headers)
        headers = worksheet.row_values(1)
        return headers
        
    except Exception as e:
        logger.error(f"❌ Failed to get sheet columns: {e}")
        return []


def validate_sheet_structure(sheets_client, required_columns):
    """Validate that Google Sheet has all required columns"""
    try:
        actual_columns = get_sheet_columns(sheets_client)
        
        # Convert to lowercase for comparison
        actual_lower = [col.lower().replace(' ', '_') for col in actual_columns]
        required_lower = [col.lower() for col in required_columns]
        
        missing = [col for col in required_lower if col not in actual_lower]
        
        if missing:
            logger.warning(f"⚠️ Missing columns in Google Sheet: {missing}")
            return False
        
        logger.info("✅ Sheet structure validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to validate sheet structure: {e}")
        return False
