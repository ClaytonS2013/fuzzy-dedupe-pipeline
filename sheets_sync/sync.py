"""
Google Sheets and Supabase synchronization module
Handles bidirectional sync between Google Sheets and Supabase
"""

import os
import logging
from typing import Dict, List, Any
import traceback

logger = logging.getLogger(__name__)

def sync_sheets_to_supabase(sheets_client, supabase_client) -> int:
    """
    Sync data from Google Sheets to Supabase
    Returns number of records processed
    """
    try:
        logger.info("📥 Starting Google Sheets → Supabase sync...")
        
        # Get spreadsheet ID from environment
        spreadsheet_id = os.getenv('SPREADSHEET_ID')
        sheet_name = os.getenv('SHEET_NAME', 'Raw_Practices')
        
        # Open the spreadsheet
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # Get all records
        records = worksheet.get_all_records()
        logger.info(f"📊 Found {len(records)} rows in {sheet_name}")
        
        if not records:
            logger.warning("⚠️ No records found in sheet")
            return 0
        
        # Clear existing records in Supabase
        logger.info("🗑️ Clearing existing practice_records...")
        supabase_client.table('practice_records').delete().neq('id', 0).execute()
        
        # Prepare records for insertion
        batch_size = 50
        total_inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            
            # Transform records to match database schema
            transformed_batch = []
            for record in batch:
                transformed_record = transform_record_for_db(record)
                if transformed_record:
                    transformed_batch.append(transformed_record)
            
            if transformed_batch:
                try:
                    response = supabase_client.table('practice_records').insert(transformed_batch).execute()
                    total_inserted += len(transformed_batch)
                    logger.info(f"✅ Inserted batch {i//batch_size + 1}: {len(transformed_batch)} records")
                except Exception as e:
                    logger.error(f"❌ Failed to insert batch {i//batch_size + 1}: {str(e)}")
                    # Log sample record structure for debugging
                    if records:
                        logger.error(f"🔍 Sample record structure: {list(records[0].keys())}")
                    raise
        
        logger.info(f"✅ Sheets → Supabase sync completed: {total_inserted} records")
        return total_inserted
        
    except Exception as e:
        logger.error(f"❌ Sheets → Supabase sync failed: {str(e)}")
        raise

def transform_record_for_db(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a record from Google Sheets format to database format
    Handle column name mappings and data type conversions
    """
    try:
        transformed = {}
        
        # Direct mappings (all as TEXT in database now)
        column_mappings = {
            'epd#': 'epd#',
            'place_id': 'place_id',
            'url': 'url',
            'country': 'country',
            'name': 'name',
            'category': 'category',
            'address': 'address',
            'full_address': 'full_address',
            'street_address': 'street_address',
            'street_address_two': 'street_address_two',
            'city': 'city',
            'state': 'state',
            'zip': 'zip',
            'open_hours': 'open_hours',
            'reviews_count': 'reviews_count',
            'rating': 'rating',
            'main_image': 'main_image',
            'reviews': 'reviews',
            'lat': 'lat',
            'lon': 'lon',
            'open_website': 'open_website',
            'phone_number': 'phone_number',
            'permanently_closed': 'permanently_closed',
            'photos_and_videos': 'photos_and_videos',
            'cid_location': 'cid_location',
            'is_claimed': 'is_claimed',
            'fid_location': 'fid_location',
            'review_distribution': 'review_distribution',
            'clay_changed': 'clay_changed',
            'change_needed': 'change_needed',
            'confidence': 'confidence',
            'status': 'status',
            'practice_type': 'practice_type',
            'suggested_fix': 'suggested_fix',
            'canonical': 'canonical',
            'reasoning': 'reasoning',
            'cluster_id': 'cluster_id'
        }
        
        for sheet_col, db_col in column_mappings.items():
            if sheet_col in record:
                value = record[sheet_col]
                # Convert everything to string or None
                if value is None or value == '' or str(value).lower() in ['nan', 'none']:
                    transformed[db_col] = None
                else:
                    # Everything is TEXT in the database now
                    transformed[db_col] = str(value)
        
        return transformed
        
    except Exception as e:
        logger.error(f"❌ Failed to transform record: {str(e)}")
        logger.error(f"   Record: {record}")
        return None

def sync_supabase_to_sheets(supabase_client, sheets_client) -> int:
    """
    Sync dedupe results from Supabase back to Google Sheets
    Returns number of records written
    """
    try:
        logger.info("📤 Starting Supabase → Google Sheets writeback...")
        
        # Fetch dedupe results
        logger.info("🔍 Fetching dedupe_results from Supabase...")
        response = supabase_client.table('dedupe_results').select("*").execute()
        results = response.data
        
        logger.info(f"📊 Found {len(results)} dedupe results")
        
        if not results:
            logger.warning("⚠️ No dedupe results found")
            return 0
        
        # Get spreadsheet
        spreadsheet_id = os.getenv('SPREADSHEET_ID')
        spreadsheet = sheets_client.open_by_key(spreadsheet_id)
        
        # Get or create Clean Data worksheet
        try:
            worksheet = spreadsheet.worksheet('Clean Data')
            logger.info("🗑️ Clearing Clean Data worksheet...")
            worksheet.clear()
        except:
            logger.info("📝 Creating Clean Data worksheet...")
            worksheet = spreadsheet.add_worksheet(title='Clean Data', rows=1000, cols=20)
        
        # Prepare headers
        headers = ['id', 'name', 'address', 'city', 'state', 'zip', 
                  'phone', 'email', 'website', 'cluster_id', 'confidence_score', 'duplicate_count']
        
        # Prepare data rows
        data_rows = [headers]
        for result in results:
            row = []
            for header in headers:
                value = result.get(header, '')
                # Convert None to empty string for Sheets
                row.append('' if value is None else str(value))
            data_rows.append(row)
        
        # Write to sheet
        logger.info(f"✍️ Writing {len(data_rows)} rows to Clean Data worksheet...")
        worksheet.update('A1', data_rows)
        
        logger.info(f"✅ Supabase → Google Sheets writeback completed: {len(results)} records")
        return len(results)
        
    except Exception as e:
        logger.error(f"❌ Supabase → Google Sheets writeback failed: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise
