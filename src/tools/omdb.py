"""
src/tools/omdb.py
─────────────────
Tool 2: Generate a dark-themed episode ratings chart using OMDB API.
"""

from __future__ import annotations

import time

import matplotlib
import matplotlib.pyplot as plt
import requests
from langchain_core.tools import tool

from config.settings import settings

matplotlib.use("Agg")


@tool
def omdb_graph_generator(anime_title: str, season: int = 1) -> str:
    """Generate a dark-themed episode ratings chart for an anime season.
    Use this when the user asks about episode ratings or the highest rated episode.
    IMPORTANT: Do NOT use markdown image syntax (![]) in your response. The UI will display the image automatically.
    Args:
        anime_title: English anime name.
        season: Season number (default 1).
    """
    try:
        r1 = requests.get(
            "https://www.omdbapi.com/",
            params={
                "apikey": settings.OMDB_API_KEY,
                "t": anime_title,
                "type": "series",
            },
            timeout=10,
        )
        r1.raise_for_status()
        s = r1.json()

        if s.get("Response") == "False":
            return f'Not found on OMDB: "{anime_title}". Try the English title.'

        imdb_id = s["imdbID"]
        exact_title = s.get("Title", anime_title)
        time.sleep(1.5)

        r2 = requests.get(
            "https://www.omdbapi.com/",
            params={
                "apikey": settings.OMDB_API_KEY,
                "i": imdb_id,
                "Season": season,
            },
            timeout=10,
        )
        r2.raise_for_status()
        eps = r2.json().get("Episodes", [])

        if not eps:
            return f"No episodes found for {exact_title} Season {season}."

        ep_nums, ratings = [], []
        for ep in eps:
            r = ep.get("imdbRating", "N/A")
            if r != "N/A":
                ep_nums.append(int(ep.get("Episode", 0)))
                ratings.append(float(r))

        if not ratings:
            return f"No ratings yet for {exact_title} S{season}."

        # ── Dark-themed chart ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(
            ep_nums, ratings,
            marker="o", linewidth=2.5, color="#E63946",
            markersize=7, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor="#E63946",
        )
        ax.fill_between(ep_nums, ratings, alpha=0.08, color="#E63946")

        bi = ratings.index(max(ratings))
        ax.annotate(
            f"Best\nEp {ep_nums[bi]}: {ratings[bi]}★",
            xy=(ep_nums[bi], ratings[bi]),
            xytext=(ep_nums[bi] + 0.8, ratings[bi] - 0.4),
            fontsize=9, color="#E63946",
            arrowprops=dict(arrowstyle="->", color="#E63946"),
        )
        ax.set_title(
            f"{exact_title} — Season {season} Ratings",
            fontsize=14, fontweight="bold", color="white",
        )
        ax.set_xlabel("Episode", color="white")
        ax.set_ylabel("IMDB Rating", color="white")
        ax.set_ylim(max(0, min(ratings) - 0.6), min(10.2, max(ratings) + 0.4))
        ax.set_facecolor("#0f0f0f")
        fig.patch.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#cccccc")
        ax.grid(axis="y", linestyle="--", alpha=0.25, color="gray")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        plt.tight_layout()

        settings.ensure_dirs()
        import re
        safe = re.sub(r'[\\/*?:"<>|]', "", exact_title).replace(" ", "_")
        path = settings.CHARTS_DIR / f"{safe}_S{season}_ratings.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        avg = round(sum(ratings) / len(ratings), 2)
        return (
            f"Chart saved: {path}\n"
            f"Episodes: {len(ratings)}\n"
            f"Average: {avg}\n"
            f"Best: Ep {ep_nums[bi]} ({ratings[bi]}★)"
        )

    except Exception as e:
        return f"Ratings lookup error: {e}"
