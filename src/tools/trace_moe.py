"""
src/tools/trace_moe.py
──────────────────────
Tool 1: Identify which anime episode a screenshot is from using trace.moe.
"""

from __future__ import annotations

import os
import time

import requests
from langchain_core.tools import tool


@tool
def trace_moe_vision(image_path: str) -> str:
    """Identify which anime episode a screenshot is from using trace.moe.
    Use this when the user uploads an image and wants to know what anime
    or episode it is from.
    Args:
        image_path: Local file path to the screenshot (jpg, png, webp)
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        ext = image_path.lower().rsplit(".", 1)[-1]
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        response = requests.post(
            "https://api.trace.moe/search?anilistInfo",
            files={"image": (os.path.basename(image_path), image_data, mime_type)},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("result"):
            return "Could not identify the anime. Try a clearer screenshot."

        top = data["result"][0]
        similarity = round(top.get("similarity", 0) * 100, 1)

        if similarity < 70:
            return (
                f"Low confidence ({similarity}%). "
                "Try a screenshot with a character clearly visible."
            )

        anilist = top.get("anilist", {})
        titles = anilist.get("title", {})
        title = titles.get("english") or titles.get("romaji") or "Unknown"
        episode = top.get("episode", "Unknown")
        ts = top.get("from", 0)

        time.sleep(1.5)  # respect trace.moe rate limit
        return (
            f"Anime: {title}\n"
            f"Episode: {episode}\n"
            f"Timestamp: {int(ts // 60)}m {int(ts % 60)}s\n"
            f"Confidence: {similarity}%"
        )

    except FileNotFoundError:
        return f"Image not found: {image_path}"
    except Exception as e:
        return f"Screenshot lookup error: {e}"
