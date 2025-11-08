#!/usr/bin/env python3
"""
Environment Validation and Auto-Fix Script
Run this BEFORE main.py to ensure everything is properly configured
"""

import os
import sys
import json
import base64
import subprocess

def check_and_install_packages():
    """Ensure all required packages are installed"""
    required_packages = [
        'supabase',
        'gspread',
        'google-auth',
        'google-auth-oauthlib',
        'google-auth-httplib2',
        'pandas',
        'numpy',
        'python-dotenv',
        'requests'
    ]
    
    print("📦 Checking required packages...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"⚠️ {package} not found, installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")

def validate_environment():
    """Validate all environment variables"""
    errors = []
    warnings = []
    
    print("\n🔍 Validating environment variables...")
    
    # Check GOOGLE_CREDENTIALS
    google_creds = os.environ.get('GOOGLE_CREDENTIALS')
    if not google_creds:
        errors.append("GOOGLE_CREDENTIALS is not set")
    else:
        print(f"✅ GOOGLE_CREDENTIALS is set ({len(google_creds)} chars)")
        
        # Try to decode it
        try:
            if google_creds.startswith('{'):
                # It's JSON
                creds_dict = json.loads(google_creds)
                print("   Format: Direct JSON")
            else:
                # It's base64
                decoded = base64.b64decode(google_creds).decode('utf-8')
                creds_dict = json.loads(decoded)
                print("   Format: Base64 encoded")
                
            # Validate required fields
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            for field in required_fields:
                if field not in creds_dict:
                    warnings.append(f"Missing field in Google credentials: {field}")
                else:
                    print(f"   ✓ Has {field}")
                    
        except Exception as e:
            errors.append(f"Cannot parse GOOGLE_CREDENTIALS: {e}")
    
    # Check SUPABASE_URL
    supabase_url = os.environ.get('SUPABASE_URL')
    if not supabase_url:
        errors.append("SUPABASE_URL is not set")
    elif not supabase_url.startswith('https://'):
        warnings.append(f"SUPABASE_URL should start with https:// (got: {supabase_url[:20]}...)")
    else:
        print(f"✅ SUPABASE_URL: {supabase_url}")
    
    # Check SUPABASE_KEY
    supabase_key = os.environ.get('SUPABASE_KEY')
    if not supabase_key:
        errors.append("SUPABASE_KEY is not set")
    elif len(supabase_key) < 100:
        warnings.append(f"SUPABASE_KEY seems too short ({len(supabase_key)} chars)")
    else:
        print(f"✅ SUPABASE_KEY is set ({len(supabase_key)} chars)")
    
    # Check SPREADSHEET_ID
    spreadsheet_id = os.environ.get('SPREADSHEET_ID')
    if not spreadsheet_id:
        errors.append("SPREADSHEET_ID is not set")
    else:
        print(f"✅ SPREADSHEET_ID: {spreadsheet_id}")
    
    # Optional variables
    optional_vars = {
        'SOURCE_TABLE': 'practice_records',
        'RESULTS_TABLE': 'dedupe_results',
        'LOG_TABLE': 'dedupe_log',
        'THRESHOLD': '75',
        'BATCH_SIZE': '100'
    }
    
    print("\n📋 Optional variables:")
    for var, default in optional_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"ℹ️ {var}: not set (will use default: {default})")
    
    return errors, warnings

def test_imports():
    """Test that all imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        from supabase import create_client
        print("✅ supabase imports correctly")
    except ImportError as e:
        print(f"❌ Cannot import supabase: {e}")
        return False
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        print("✅ Google libraries import correctly")
    except ImportError as e:
        print(f"❌ Cannot import Google libraries: {e}")
        return False
    
    try:
        from sheets_sync.sync import sync_sheets_to_supabase, sync_supabase_to_sheets
        print("✅ sheets_sync module imports correctly")
    except ImportError as e:
        print(f"⚠️ Cannot import sheets_sync: {e}")
    
    try:
        from dedupe_logic.processor import run_deduplication
        print("✅ dedupe_logic module imports correctly")
    except ImportError as e:
        print(f"⚠️ Cannot import dedupe_logic: {e}")
    
    return True

def main():
    """Main validation function"""
    print("=" * 60)
    print("🔧 FUZZY DEDUPE PIPELINE - ENVIRONMENT VALIDATOR")
    print("=" * 60)
    
    # Check packages
    check_and_install_packages()
    
    # Validate environment
    errors, warnings = validate_environment()
    
    # Test imports
    imports_ok = test_imports()
    
    # Report results
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    
    if errors:
        print("\n❌ ERRORS (must fix):")
        for error in errors:
            print(f"   • {error}")
    
    if warnings:
        print("\n⚠️ WARNINGS (should review):")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not errors and imports_ok:
        print("\n✅ Environment is ready! You can run the pipeline.")
        return 0
    else:
        print("\n❌ Environment has issues. Please fix the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
