"""
Utility Functions Module
Helper functions for text processing, cleaning, and formatting.
"""

import re
import logging
import os
from typing import Any, Dict, List
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text (str): Raw text content
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Clean up common PDF artifacts
    text = re.sub(r'[\x0c\x0b\x07]', ' ', text)  # Form feed, vertical tab, bell
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_bullets(text: str) -> str:
    """
    Normalize bullet points and list formatting.
    
    Args:
        text (str): Text with bullet points
        
    Returns:
        str: Normalized text
    """
    # Replace various bullet characters with standard bullet
    bullet_chars = ['•', '◦', '▪', '▫', '‣', '⁃']
    for char in bullet_chars:
        text = text.replace(char, '•')
    
    # Normalize numbered lists
    text = re.sub(r'^(\d+)[.)]\s+', r'\1. ', text, flags=re.MULTILINE)
    
    # Normalize lettered lists
    text = re.sub(r'^([a-zA-Z])[.)]\s+', r'\1. ', text, flags=re.MULTILINE)
    
    return text


def format_citation(page: int, table_id: str = None, section: str = None) -> str:
    """
    Format citation information for source references.
    
    Args:
        page (int): Page number
        table_id (str, optional): Table identifier
        section (str, optional): Section name
        
    Returns:
        str: Formatted citation
    """
    citation_parts = [f"Page {page}"]
    
    if table_id:
        citation_parts.append(f"Table {table_id}")
    
    if section and section != "General":
        citation_parts.append(f"Section: {section}")
    
    return f"Source: {', '.join(citation_parts)}"


def format_table_for_display(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Format DataFrame for readable display in chat interface.
    
    Args:
        df (pd.DataFrame): Table data
        max_rows (int): Maximum rows to display
        
    Returns:
        str: Formatted table string
    """
    if df.empty:
        return "Empty table"
    
    # Limit rows for display
    display_df = df.head(max_rows)
    
    # Create a simple text representation
    lines = []
    
    # Add headers
    headers = " | ".join(str(col)[:20] for col in display_df.columns)
    lines.append(headers)
    lines.append("-" * len(headers))
    
    # Add data rows
    for _, row in display_df.iterrows():
        row_text = " | ".join(str(val)[:20] for val in row.values)
        lines.append(row_text)
    
    # Add indication if table was truncated
    if len(df) > max_rows:
        lines.append(f"... ({len(df) - max_rows} more rows)")
    
    return "\n".join(lines)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for better search relevance.
    
    Args:
        text (str): Input text
        max_keywords (int): Maximum number of keywords
        
    Returns:
        List[str]: List of keywords
    """
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall'
    }
    
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:max_keywords]]


def validate_file_path(file_path: str) -> bool:
    """
    Validate if file path exists and is accessible.
    
    Args:
        file_path (str): Path to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        return os.path.exists(file_path) and os.path.isfile(file_path)
    except Exception:
        return False


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing problematic characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Safe filename
    """
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_{2,}', '_', filename)
    
    # Trim underscores from start and end
    filename = filename.strip('_')
    
    return filename


def chunk_text_by_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks by sentences, respecting size limits.
    
    Args:
        text (str): Text to chunk
        max_chunk_size (int): Maximum characters per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def create_progress_message(current: int, total: int, operation: str) -> str:
    """
    Create a progress message for long operations.
    
    Args:
        current (int): Current progress
        total (int): Total items
        operation (str): Operation description
        
    Returns:
        str: Progress message
    """
    percentage = (current / total) * 100 if total > 0 else 0
    return f"{operation}: {current}/{total} ({percentage:.1f}%)"


def log_operation(operation: str, success: bool, details: str = "") -> None:
    """
    Log operation results consistently.
    
    Args:
        operation (str): Operation name
        success (bool): Whether operation succeeded
        details (str): Additional details
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"{operation} - {status}"
    
    if details:
        message += f": {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)
