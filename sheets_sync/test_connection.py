#!/usr/bin/env python3
"""
Connection Test Script
Tests Supabase and Google Sheets connections before deployment
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test that all required environment variables are set."""
    print("\n" + "="*60)
    print("TEST 1: Environment Variables")
    print("="*60)
    
    required_vars = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "SPREADSHEET_ID": os.getenv("SPREADSHEET_ID"),
        "TIMEZONE": os.getenv("TIMEZONE"),
    }
    
    # Check for either local file or base64
    has_local_creds = os.path.exists(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service_account.json"))
    has_base64_creds = bool(os.getenv("GOOGLE_CREDENTIALS_BASE64"))
    
    all_set = True
    for var, value in required_vars.items():
        if value:
            print(f"\u2713 {var}: {value[:30]}..." if len(value) > 30 else f"\u2713 {var}: {value}")
        else:
            print(f"\u2717 {var}: NOT SET")
            all_set = False
    
    if has_local_creds:
        print(f"\u2713 GOOGLE_APPLICATION_CREDENTIALS: File exists")
    elif has_base64_creds:
        print(f"\u2713 GOOGLE_CREDENTIALS_BASE64: Set")
    else:
        print(f"\u2717 Google credentials: NEITHER file nor base64 set")
        all_set = False
    
    if all_set:
        print("\n\u2705 All environment variables are set!")
        return True
    else:
        print("\n\u274c Some environment variables are missing!")
        return False


def test_supabase_connection():
    """Test connection to Supabase."""
    print("\n" + "="*60)
    print("TEST 2: Supabase Connection")
    print("="*60)
    
    try:
        import requests
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("\u274c Supabase credentials not set")
            return False
        
        # Test basic connection
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }
        
        print("Testing connection to Supabase...")
        response = requests.get(
            f"{supabase_url}/rest/v1/",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("\u2713 Supabase API is reachable")
        else:
            print(f"\u2717 Supabase API returned status {response.status_code}")
            return False
        
        # Test sync_state table
        print("\nTesting sync_state table...")
        response = requests.get(
            f"{supabase_url}/rest/v1/sync_state",
            headers=headers,
            params={"id": "eq.1"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print("\u2713 sync_state table exists and is accessible")
                print(f"  Last marker: {data[0].get('last_processed_marker', 'None')[:20]}...")
                print(f"  Last run: {data[0].get('last_run_at', 'Never')}")
            else:
                print("\u26a0 sync_state table exists but is empty")
                print("  Run the SQL setup script to initialize it")
        else:
            print(f"\u2717 sync_state table not accessible (status {response.status_code})")
            print("  Run the setup.sql script in Supabase SQL Editor")
            return False
        
        # Test practice_records table
        print("\nTesting practice_records table...")
        response = requests.get(
            f"{supabase_url}/rest/v1/practice_records",
            headers=headers,
            params={"limit": "1"},
            timeout=10
        )
        
        if response.status_code == 200:
            print("\u2713 practice_records table exists and is accessible")
            data = response.json()
            print(f"  Current records: {len(data)} (showing first 1)")
        else:
            print(f"\u2717 practice_records table not accessible (status {response.status_code})")
            return False
        
        # Test dedupe_results table
        print("\nTesting dedupe_results table...")
        response = requests.get(
            f"{supabase_url}/rest/v1/dedupe_results",
            headers=headers,
            params={"limit": "1"},
            timeout=10
        )
        
        if response.status_code == 200:
            print("\u2713 dedupe_results table exists and is accessible")
            data = response.json()
            print(f"  Current records: {len(data)} (showing first 1)")
        else:
            print(f"\u2717 dedupe_results table not accessible (status {response.status_code})")
            return False
        
        print("\n\u2705 Supabase connection successful!")
        return True
        
    except ImportError:
        print("\u274c 'requests' library not installed")
        print("   Run: pip install requests")
        return False
    except Exception as e:
        print(f"\u274c Error testing Supabase: {e}")
        return False


def test_google_sheets_connection():
    """Test connection to Google Sheets."""
    print("\n" + "="*60)
    print("TEST 3: Google Sheets Connection")
    print("="*60)
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        spreadsheet_id = os.getenv("SPREADSHEET_ID")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./service_account.json")
        
        if not spreadsheet_id:
            print("\u274c SPREADSHEET_ID not set")
            return False
        
        if not os.path.exists(creds_path):
            print(f"\u274c Service account file not found: {creds_path}")
            return False
        
        print("Authenticating with Google Sheets API...")
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        client = gspread.authorize(creds)
        
        print("\u2713 Google Sheets authentication successful")
        
        print(f"\nOpening spreadsheet (ID: {spreadsheet_id[:20]}...)...")
        sheet = client.open_by_key(spreadsheet_id)
        print(f"\u2713 Spreadsheet opened: '{sheet.title}'")
        
        # Test Raw_Practices worksheet
        print("\nTesting Raw_Practices worksheet...")
        try:
            worksheet = sheet.worksheet("Raw_Practices")
            print(f"\u2713 Raw_Practices worksheet found")
            
            values = worksheet.get_all_values()
            if values:
                print(f"  Rows: {len(values)}")
                print(f"  Headers: {values[0] if values else 'None'}")
                if len(values) > 1:
                    print(f"  Data rows: {len(values) - 1}")
                else:
                    print("  \u26a0 No data rows (only headers)")
            else:
                print("  \u26a0 Worksheet is empty")
        except gspread.exceptions.WorksheetNotFound:
            print("\u2717 Raw_Practices worksheet not found")
            print("  Create a worksheet named 'Raw_Practices' in your spreadsheet")
            return False
        
        # Test Clean_Data worksheet
        print("\nTesting Clean_Data worksheet...")
        try:
            worksheet = sheet.worksheet("Clean_Data")
            print(f"\u2713 Clean_Data worksheet found")
            
            values = worksheet.get_all_values()
            if values:
                print(f"  Rows: {len(values)}")
            else:
                print("  Worksheet is empty (will be populated on first sync)")
        except gspread.exceptions.WorksheetNotFound:
            print("\u2717 Clean_Data worksheet not found")
            print("  Create a worksheet named 'Clean_Data' in your spreadsheet")
            return False
        
        print("\n\u2705 Google Sheets connection successful!")
        return True
        
    except ImportError as e:
        print(f"\u274c Required library not installed: {e}")
        print("   Run: pip install gspread google-auth")
        return False
    except Exception as e:
        print(f"\u274c Error testing Google Sheets: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SHEETS-SUPABASE SYNC - CONNECTION TEST")
    print("="*60)
    
    results = {
        "Environment Variables": test_environment_variables(),
        "Supabase Connection": test_supabase_connection(),
        "Google Sheets Connection": test_google_sheets_connection(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "\u2705 PASSED" if passed else "\u274c FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("\U0001f389 ALL TESTS PASSED!")
        print("="*60)
        print("You're ready to deploy to Railway!")
        print("\nNext steps:")
        print("1. Convert service_account.json to base64 for Railway")
        print("2. Push code to GitHub")
        print("3. Deploy to Railway with environment variables")
        return 0
    else:
        print("\n" + "="*60)
        print("\u26a0\ufe0f  SOME TESTS FAILED")
        print("="*60)
        print("Fix the issues above before deploying.")
        print("\nCommon fixes:")
        print("- Set missing environment variables in .env file")
        print("- Run setup.sql in Supabase SQL Editor")
        print("- Create missing worksheets in Google Sheets")
        print("- Verify service account email has access to the sheet")
        return 1


if __name__ == "__main__":
    sys.exit(main())
