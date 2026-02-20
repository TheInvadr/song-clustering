from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
import urllib.parse

import requests


CACHE_PATH = Path("cache/previews_itunes.json")


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _key(song_title: str, artist: str) -> str:
    return f"{song_title.strip().lower()}|||{artist.strip().lower()}"


def fetch_itunes_preview_url(song_title: str, artist: str, timeout: int = 10) -> Optional[str]:

    cache = _load_cache()
    k = _key(song_title, artist)
    if k in cache:
        return cache[k] or None

    query = f"{song_title} {artist}"
    params = {
        "term": query,
        "entity": "song",
        "limit": 5,
    }
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(params)

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        cache[k] = None
        _save_cache(cache)
        return None

    results = data.get("results", [])
    preview = None
    if results:
        for item in results:
            p = item.get("previewUrl")
            if p:
                preview = p
                break

    cache[k] = preview
    _save_cache(cache)
    return preview
