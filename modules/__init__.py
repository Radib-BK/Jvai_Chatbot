"""
Financial Policy Chatbot Modules Package
Modular components for AI-powered document analysis and conversational search.
"""

from .pdf_extractor import PDFExtractor
from .chunker import ContentChunker
from .embedding_index import EmbeddingIndex
from .searcher import ContentSearcher
from .answerer import ConversationalAnswerer
from .utils import (
    clean_text,
    normalize_bullets,
    format_citation,
    format_table_for_display,
    truncate_text,
    extract_keywords,
    validate_file_path,
    safe_filename,
    log_operation
)

__version__ = "1.0.0"
__author__ = "Financial Policy Chatbot Team"

__all__ = [
    "PDFExtractor",
    "ContentChunker", 
    "EmbeddingIndex",
    "ContentSearcher",
    "ConversationalAnswerer",
    "clean_text",
    "normalize_bullets",
    "format_citation",
    "format_table_for_display",
    "truncate_text",
    "extract_keywords",
    "validate_file_path",
    "safe_filename",
    "log_operation"
]
