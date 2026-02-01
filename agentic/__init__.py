# Agentic RAG components for TextGuard
# This package provides advanced retrieval capabilities

from .router import get_agentic_router
from .indexer import AdvancedIndexer, get_query_transformer
from .retriever import IterativeRetriever
from .evaluator import SelfEvaluator

__all__ = [
    "get_agentic_router",
    "get_query_transformer",
    "AdvancedIndexer",
    "IterativeRetriever",
    "SelfEvaluator"
]

__version__ = "0.1.0"

__author__ = "Kerman"
