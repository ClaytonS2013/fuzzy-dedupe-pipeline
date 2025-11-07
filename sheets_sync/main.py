"""
sheets_sync/main.py

Bidirectional sync between Google Sheets and Supabase
for the Fuzzy Matching Antelligence pipeline.
"""

import os
import logging
from typing import List, Dict, Any

import gspread
from google.oauth2.service_account import Credentials
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SheetsSupabaseSync:
    """Handles bidirectional sync between Google Sheets and Supabase."""
    
    def __init__(self):
        """Initialize with environment variables."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.spreadsheet_id = os.getenv("SPREADSHEET_ID")
        self.creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/service_account.json")
        
        # Validate configuration
        if not all([self.supabase_url, self.supabase_key, self.spreadsheet_id]):
            raise ValueError(
                "Missing required environment variables: "
                "SUPABASE_URL, SUPABASE_KEY, or SPREADSHEET_ID"
            )
        
        # Initialize clients
        self.sheets_client = self._init_sheets_client()
        self.supabase_headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        # Normalization configuration
        # These functions normalize specific fields before insertion.
        # Names and addresses are lower‑cased and stripped of extra whitespace and punctuation.
        # Phone numbers are stripped of non‑digit characters. Emails are lower‑cased.
        # You can override these behaviours by subclassing or updating these methods.
        # Note: heavy libraries like `phonenumbers` are optional and may not be installed in all environments.
        try:
            import phonenumbers  # type: ignore
            self._phonenumbers_available = True
            self._phonenumbers = phonenumbers
        except Exception:
            self._phonenumbers_available = False
            self._phonenumbers = None
    
    def _init_sheets_client(self) -> gspread.Client:
        """Initialize Google Sheets client."""
        try:
            if not os.path.exists(self.creds_path):
                raise FileNotFoundError(f"Service account file not found: {self.creds_path}")
            
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            creds = Credentials.from_service_account_file(self.creds_path, scopes=scopes)
            client = gspread.authorize(creds)
            logger.info("✅ Google Sheets client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Sheets client: {e}")
            raise

    # ---------------------------------------------------------------------------
    # Normalization helpers
    #
    # The following helper methods provide simple normalization for names, addresses,
    # phone numbers and emails. These are used when importing data from Google
    # Sheets so that your deduplication logic can operate on more consistent
    # representations. Feel free to extend these methods or override them via
    # subclassing if you need more sophisticated normalisation.

    def _normalize_name(self, value: str) -> str:
        """Normalize a name string by lowercasing and stripping punctuation."""
        if not isinstance(value, str):
            return value
        import re
        value = value.strip().lower()
        # Remove punctuation characters
        value = re.sub(r"[\p{P}\p{S}]", "", value)
        # Collapse multiple spaces
        value = re.sub(r"\s+", " ", value)
        return value

    def _normalize_address(self, value: str) -> str:
        """Normalize an address string: lower-case, standardise common abbreviations."""
        if not isinstance(value, str):
            return value
        value = value.strip().lower()
        # Replace common abbreviations
        replacements = {
            " st ": " street ",
            " rd ": " road ",
            " ave ": " avenue ",
            " blvd ": " boulevard ",
            " dr ": " drive ",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        value = " ".join(value.split())
        return value

    def _normalize_phone(self, value: str) -> str:
        """Normalize a phone number by stripping non‑digits and formatting."""
        if not isinstance(value, str):
            return value
        # Remove all non-digit characters
        digits = ''.join(filter(str.isdigit, value))
        if not digits:
            return ''
        # If phonenumbers library is available, attempt to format
        if self._phonenumbers_available:
            try:
                parsed = self._phonenumbers.parse(digits, None)
                if self._phonenumbers.is_valid_number(parsed):
                    return self._phonenumbers.format_number(parsed, self._phonenumbers.PhoneNumberFormat.E164)
            except Exception:
                pass
        # Fallback: return the digits as is
        return digits

    def _normalize_email(self, value: str) -> str:
        """Normalize an email by lowercasing and stripping whitespace."""
        if not isinstance(value, str):
            return value
        return value.strip().lower()

    def _normalize_field(self, header: str, value: str) -> str:
        """Normalize a field based on header name."""
        header_lower = header.lower()
        if 'name' in header_lower:
            return self._normalize_name(value)
        if 'address' in header_lower:
            return self._normalize_address(value)
        if 'phone' in header_lower:
            return self._normalize_phone(value)
        if 'email' in header_lower:
            return self._normalize_email(value)
        return value
    
    def _supabase_request(self, method: str, endpoint: str, data: Any = None) -> requests.Response:
        """Make a request to Supabase REST API."""
        url = f"{self.supabase_url}/rest/v1/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.supabase_headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=self.supabase_headers, json=data, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.supabase_headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Supabase API request failed: {e}")
            raise
    
    def sync_sheets_to_supabase(self) -> int:
        """
        Sync data from Google Sheets 'Raw_Practices' to Supabase 'practice_records'.
        
        Returns:
            Number of rows synced
        """
        try:
            logger.info("📥 Starting Google Sheets → Supabase sync...")
            
            # Open spreadsheet and get Raw_Practices worksheet
            sheet = self.sheets_client.open_by_key(self.spreadsheet_id)
            worksheet = sheet.worksheet("Raw_Practices")
            
            # Get all values
            all_values = worksheet.get_all_values()
            
            if not all_values:
                logger.warning("⚠️ Raw_Practices worksheet is empty")
                return 0
            
            # First row is headers
            headers = all_values[0]
            data_rows = all_values[1:]
            
            if not data_rows:
                logger.warning("⚠️ No data rows in Raw_Practices (only headers)")
                return 0
            
            logger.info(f"📊 Found {len(data_rows)} rows in Raw_Practices")
            
            # Convert rows to dictionaries
            records = []
            for idx, row in enumerate(data_rows):
                # Pad row to match header length
                padded_row = row + [''] * (len(headers) - len(row))
                
                record: Dict[str, Any] = {}
                for i, header in enumerate(headers):
                    # Clean header name (lowercase, replace spaces with underscores)
                    clean_header = header.lower().replace(' ', '_').replace('-', '_')
                    original_value = padded_row[i].strip() if i < len(padded_row) else ''
                    record[clean_header] = original_value
                    # Store a normalised version for fields likely used in deduplication
                    try:
                        normalized_value = self._normalize_field(header, original_value)
                    except Exception:
                        normalized_value = original_value
                    # Append suffix _norm to avoid clashing with original fields
                    record[f"{clean_header}_norm"] = normalized_value
                records.append(record)
            
            # Clear existing data in practice_records
            logger.info("🗑️ Clearing existing practice_records...")
            try:
                self._supabase_request("DELETE", "practice_records?id=gte.0")
            except Exception as e:
                logger.warning(f"⚠️ Could not clear practice_records (may not exist or be empty): {e}")
            
            # Insert records in batches
            batch_size = 500
            total_inserted = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                try:
                    response = self._supabase_request("POST", "practice_records", data=batch)
                    result = response.json()
                    inserted_count = len(result) if isinstance(result, list) else 1
                    total_inserted += inserted_count
                    logger.info(f"✅ Inserted batch {i // batch_size + 1}: {inserted_count} records")
                except Exception as e:
                    logger.error(f"❌ Failed to insert batch {i // batch_size + 1}: {e}")
                    raise
            
            logger.info(f"✅ Google Sheets → Supabase sync completed: {total_inserted} records")
            return total_inserted
            
        except gspread.exceptions.WorksheetNotFound:
            logger.error("❌ Worksheet 'Raw_Practices' not found in spreadsheet")
            raise
        except Exception as e:
            logger.error(f"❌ Sheets → Supabase sync failed: {e}")
            raise
    
    def write_clean_data_to_sheet(self) -> int:
        """
        Write cleaned data from Supabase 'dedupe_results' to Google Sheets 'Clean Data'.
        
        Returns:
            Number of rows written
        """
        try:
            logger.info("📤 Starting Supabase → Google Sheets writeback...")
            
            # Fetch dedupe results from Supabase
            logger.info("🔍 Fetching dedupe_results from Supabase...")
            response = self._supabase_request("GET", "dedupe_results?select=*&order=id.asc")
            dedupe_results = response.json()
            
            if not dedupe_results:
                logger.warning("⚠️ No dedupe results found in Supabase")
                return 0
            
            logger.info(f"📊 Found {len(dedupe_results)} dedupe results")
            
            # Open spreadsheet and get Clean Data worksheet
            sheet = self.sheets_client.open_by_key(self.spreadsheet_id)
            
            try:
                worksheet = sheet.worksheet("Clean Data")
            except gspread.exceptions.WorksheetNotFound:
                logger.info("📝 'Clean Data' worksheet not found, creating it...")
                worksheet = sheet.add_worksheet(title="Clean Data", rows=1000, cols=20)
            
            # Clear existing data
            logger.info("🗑️ Clearing Clean Data worksheet...")
            worksheet.clear()
            
            # Extract headers from first record
            headers = list(dedupe_results[0].keys())
            
            # Convert records to rows
            rows = [headers]
            for record in dedupe_results:
                row = [str(record.get(h, '')) for h in headers]
                rows.append(row)
            
            # Write to sheet
            logger.info(f"✍️ Writing {len(rows)} rows to Clean Data worksheet...")
            worksheet.update(rows, "A1")
            
            logger.info(f"✅ Supabase → Google Sheets writeback completed: {len(dedupe_results)} records")
            return len(dedupe_results)
            
        except Exception as e:
            logger.error(f"❌ Supabase → Sheets writeback failed: {e}")
            raise


# Module-level convenience functions
_sync = None

def _get_sync():
    """Get or create sync instance."""
    global _sync
    if _sync is None:
        _sync = SheetsSupabaseSync()
    return _sync


def sync_sheets_to_supabase() -> int:
    """
    Sync data from Google Sheets to Supabase.
    
    Returns:
        Number of rows synced
    """
    return _get_sync().sync_sheets_to_supabase()


def write_clean_data_to_sheet() -> int:
    """
    Write cleaned data from Supabase to Google Sheets.
    
    Returns:
        Number of rows written
    """
    return _get_sync().write_clean_data_to_sheet()


if __name__ == "__main__":
    # Test the sync functions
    print("🧪 Testing Sheets ↔ Supabase Sync")
    
    try:
        # Test Sheets → Supabase
        count = sync_sheets_to_supabase()
        print(f"✅ Synced {count} rows to Supabase")
        
        # Test Supabase → Sheets
        count = write_clean_data_to_sheet()
        print(f"✅ Wrote {count} rows to Google Sheets")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
