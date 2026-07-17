"""
Helper script to generate the base64 encoded string of token.json.
This is needed for deploying the application since token.json should not be committed to version control.
Run this script and copy the output string, then set it as the CALENDAR_TOKEN_B64 environment variable on your deployment platform.
"""

import base64
from pathlib import Path

def get_b64_token():
    token_path = Path(__file__).parent.parent / "token.json"
    if not token_path.exists():
        print("token.json not found! Please run calendar_auth.py first to generate it.")
        return
    
    with open(token_path, "r", encoding="utf-8") as f:
        token_content = f.read()
    
    b64_token = base64.b64encode(token_content.encode("utf-8")).decode("utf-8")
    print("\n--- BASE64 TOKEN ---")
    print(b64_token)
    print("--------------------\n")
    print("Set this value as the CALENDAR_TOKEN_B64 environment variable in your deployment platform.")

if __name__ == "__main__":
    get_b64_token()
