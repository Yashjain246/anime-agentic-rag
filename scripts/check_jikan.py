"""
Simple script to check if the Jikan API (MyAnimeList) is currently up or down.
Run this using: .\.venv\Scripts\python.exe scripts\check_jikan.py
"""
import requests
import sys

def check_jikan_status():
    url = "https://api.jikan.moe/v4/anime"
    params = {"q": "One Piece", "limit": 1}
    
    print("Checking Jikan API status...\n")
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            print("Jikan API is up and working.")
            data = response.json().get("data", [])
            if data:
                print(f"Sample response received: {data[0].get('title')}")
        elif response.status_code in (503, 504):
            print(f"Jikan API is down (HTTP {response.status_code}).")
            print("MyAnimeList is currently down or refusing connections.")
        elif response.status_code == 429:
            print("Jikan API is rate limited (HTTP 429).")
            print("You are making requests too fast. Wait a moment and try again.")
        else:
            print(f"Jikan API returned an unexpected status code: {response.status_code}")

    except requests.exceptions.Timeout:
        print("Jikan API connection timed out.")
        print("The server is taking too long to respond, likely experiencing heavy load.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Jikan API: {e}")

if __name__ == "__main__":
    check_jikan_status()
