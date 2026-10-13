"""Nominatim (OpenStreetMap) search, restricted to a Bengaluru viewbox. Free; max 1 request/second."""
from __future__ import annotations

import requests
import streamlit as st

_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS = {"User-Agent": "UrbanSense-Bengaluru-HeatLab/0.1 (student capstone project)"}


@st.cache_data(ttl=86400, show_spinner=False)
def search(query: str) -> tuple[float, float, str] | None:
    r = requests.get(_URL, headers=_HEADERS, timeout=10, params={
        "q": query, "format": "json", "limit": 1, "countrycodes": "in",
        "viewbox": "77.3,13.3,77.9,12.7", "bounded": 1})
    r.raise_for_status()
    hits = r.json()
    if not hits:
        return None
    h = hits[0]
    return float(h["lat"]), float(h["lon"]), h.get("display_name", query)