# conftest.py
"""
Pytest configuration — adds project root to sys.path so all
`from src...` and `from config...` imports resolve correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
