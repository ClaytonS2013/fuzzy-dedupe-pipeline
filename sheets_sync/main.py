#!/usr/bin/env python3
"""
Production Zapier Replacement: Daily Google Sheets → Supabase Sync with Change Detection

SETUP:
1. Install dependencies:
   pip install -r requirements.txt

2. Set environment variables (or use .env file):
   export SUPABASE_URL="https://<project-id>.supabase.co"
   export SUPABASE_KEY="<anon-key>"
   export SPREADSHEET_ID="<google-sheet-id>"
   export TIMEZONE="America/Chicago"
   export GOOGLE_CREDENTIALS_BASE64="<base64-encoded-service-account-json>"

3. Create sync_state table in Supabase (see setup.sql)

4. Run:
   python main.py               # Runs on schedule (04:00 America/Chicago)
   python main.py --run-now     # Manual immediate execution
"""

import base64
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import gspread
import pytz
import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DEDUPE_WAIT_TIME = 60  # seconds to wait for dedupe to complete


def setup_google_creds() -> None:
    """
    Write service account JSON from base64 env var if present.
    This allows Railway deployment without file uploads.
    """
    creds_base64 = os.getenv("GOOGLE_CREDENTIALS_BASE64")
    if creds_base64:
        try:
            creds_json = base64.b64decode(creds_base64).decode("utf-8")
            creds_path = "/tmp/service_account.json"
            with open(creds_path, "w") as f:
                f.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            logger.info("Service account credentials loaded from base64 env var")
        except Exception as e:
            logger.error(f"Failed to decode GOOGLE_CREDENTIALS_BASE64: {e}")
            raise


class Config:
    """Configuration from environment variables."""

    def __init__(self) -> None:
        self.supabase_url: str = os.getenv("SUPABASE_URL", "")
        self.supabase_key: str = os.getenv("SUPABASE_KEY", "")
        self.spreadsheet_id: str = os.getenv("SPREADSHEET_ID", "")
        self.google_creds_path: str = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS", "./service_account.json"
        )
        self.timezone: str = os.getenv("TIMEZONE", "America/Chicago")

        self._validate()

    def _validate(self) -> None:
        """Validate required configuration."""
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not self.supabase_key:
            raise ValueError("SUPABASE_KEY environment variable is required")
        if not self.spreadsheet_id:
            raise ValueError("SPREADSHEET_ID environment variable is required")
        if not os.path.exists(self.google_creds_path):
            raise ValueError(
                f"Google credentials file not found: {self.google_creds_path}"
            )


class SupabaseClient:
    """Client for Supabase REST API operations."""

    def __init__(self, config: Config) -> None:
        self.base_url = config.supabase_url.rstrip("/")
        self.headers = {
            "apikey": config.supabase_key,
            "Authorization": f"Bearer {config.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}/rest/v1/{endpoi
