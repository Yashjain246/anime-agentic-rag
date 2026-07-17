"""
scripts/calendar_auth.py
────────────────────────
ONE-TIME Google Calendar OAuth setup script.

Run this ONCE locally to generate token.json with a persistent refresh token.
The deployed app on Render will silently use the refresh token — no user
interaction ever needed again.

Usage:
  python scripts/calendar_auth.py

Prerequisites:
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable Google Calendar API
  3. Create OAuth 2.0 credentials (Desktop app type)
  4. Download credentials.json → place it in the project root
  5. Run this script → it will open a browser for one-time consent
  6. token.json will be created in the project root
  7. Upload token.json to Render as a secret environment variable
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from config.settings import settings

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def main():
    creds = None

    if settings.CALENDAR_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(
            str(settings.CALENDAR_TOKEN_PATH), SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing existing token...")
            creds.refresh(Request())
        else:
            if not settings.CALENDAR_CREDENTIALS_PATH.exists():
                print(
                    f"ERROR: credentials.json not found at {settings.CALENDAR_CREDENTIALS_PATH}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials"
                )
                sys.exit(1)

            print("Opening browser for one-time Google Calendar authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(settings.CALENDAR_CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(settings.CALENDAR_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"✅ token.json saved to: {settings.CALENDAR_TOKEN_PATH}")
        print("\nNext steps:")
        print("  1. Set ENABLE_CALENDAR_TOOL=true in your .env")
        print("  2. For Render deployment: add token.json content as a secret file")
    else:
        print(f"✅ Token is already valid: {settings.CALENDAR_TOKEN_PATH}")
        print(f"   Expires: {creds.expiry}")


if __name__ == "__main__":
    main()
