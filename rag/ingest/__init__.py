"""Knowledge ingestion utilities."""

from .pipeline import prepare_knowledge_base
from .review_json import prepare_review_json_knowledge_base

__all__ = ["prepare_knowledge_base", "prepare_review_json_knowledge_base"]
