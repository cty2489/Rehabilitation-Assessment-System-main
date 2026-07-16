#!/usr/bin/env python3
"""Run with: uvicorn rag.service:app --host 127.0.0.1 --port 8010."""

from rag.service import app


__all__ = ["app"]
