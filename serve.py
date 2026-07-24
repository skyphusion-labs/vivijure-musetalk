#!/usr/bin/env python3
"""Homelab HTTP entry for MuseTalk lipsync on LOCAL_FINISH_LIPSYNC_URL."""
import os

from handler import handler
from runpod_http_serve import run_serve

if __name__ == "__main__":
    run_serve(
        handler,
        service="vivijure-musetalk-finish-lipsync",
        port=int(os.environ.get("PORT", "8011") or "8011"),
    )
