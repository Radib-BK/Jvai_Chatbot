"""
Text Chunking Module
Handles intelligent splitting of text and table content for better processing.
"""

import pandas as pd
import re
from typing import List, Dict, Any
import logging
from .utils import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentChunker:
    """
    Intelligent chunking of text and table content with metadata preservation.
    
    This class handles the splitting of long text passages and large tables
    into manageable chunks while maintaining source references for citations.
    """
    
    def __init__(self, max_chunk_size: int = 500, overlap_size: int = 50):
        """
        Initialize the content chunker.
        
        Args:
            max_chunk_size (int): Maximum number of words per chunk
            overlap_size (int): Number of words to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_text_content(self, pages_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text content from pages while preserving metadata.
        
        Args:
            pages_text (List[Dict[str, Any]]): List of page text data
            
        Returns:
            List[Dict[str, Any]]: List of text chunks with metadata
        """
        chunks = []
        
        for page_data in pages_text:
            page_chunks = self._chunk_single_text(
                text=page_data["text"],
                page_num=page_data["page"],
                section=page_data["section"]
            )
            chunks.extend(page_chunks)
        
        logger.info(f"Created {len(chunks)} text chunks from {len(pages_text)} pages")
        return chunks
    
    def chunk_table_content(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk table content for better searchability.
        
        Args:
            tables (List[Dict[str, Any]]): List of table data
            
        Returns:
            List[Dict[str, Any]]: List of table chunks with metadata
        """
        chunks = []
        
        for table_data in tables:
            table_chunks = self._chunk_single_table(
                df=table_data["df"],
                page_num=table_data["page"],
                table_id=table_data["table_id"],
                summary=table_data["summary"]
            )
            chunks.extend(table_chunks)
        
        logger.info(f"Created {len(chunks)} table chunks from {len(tables)} tables")
        return chunks
    
    def chunk_all_content(self, document_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk both text and table content from document data.
        
        Args:
            document_data (Dict[str, Any]): Complete document data
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Chunked text and table data
        """
        text_chunks = self.chunk_text_content(document_data["pages_text"])
        table_chunks = self.chunk_table_content(document_data["tables"])
        
        return {
            "text_chunks": text_chunks,
            "table_chunks": table_chunks,
            "total_chunks": len(text_chunks) + len(table_chunks)
        }
    
    def _chunk_single_text(self, text: str, page_num: int, section: str) -> List[Dict[str, Any]]:
        """
        Chunk a single text passage using sentence-aware splitting.
        
        Args:
            text (str): Text content to chunk
            page_num (int): Source page number
            section (str): Source section name
            
        Returns:
            List[Dict[str, Any]]: List of text chunks
        """
        chunks = []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max size, finalize current chunk
            if current_word_count + sentence_words > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_text_chunk(
                    text=chunk_text,
                    page_num=page_num,
                    section=section,
                    chunk_id=len(chunks)
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_word_count = len(' '.join(current_chunk).split())
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_text_chunk(
                text=chunk_text,
                page_num=page_num,
                section=section,
                chunk_id=len(chunks)
            ))
        
        return chunks
    
    def _chunk_single_table(self, df: pd.DataFrame, page_num: int, 
                           table_id: str, summary: str) -> List[Dict[str, Any]]:
        """
        Chunk a single table based on row count and content size.
        
        Args:
            df (pd.DataFrame): Table data
            page_num (int): Source page number
            table_id (str): Table identifier
            summary (str): Table summary
            
        Returns:
            List[Dict[str, Any]]: List of table chunks
        """
        chunks = []
        
        if df.empty:
            return chunks
        
        # Create table header chunk
        header_chunk = self._create_table_chunk(
            content=f"Table Header: {', '.join(df.columns)}",
            page_num=page_num,
            table_id=table_id,
            chunk_type="header",
            summary=summary,
            chunk_id=0
        )
        chunks.append(header_chunk)
        
        # Create table summary chunk
        summary_chunk = self._create_table_chunk(
            content=f"Table Summary: {summary}",
            page_num=page_num,
            table_id=table_id,
            chunk_type="summary",
            summary=summary,
            chunk_id=1
        )
        chunks.append(summary_chunk)
        
        # Chunk table rows if table is large
        max_rows_per_chunk = 10
        
        for start_row in range(0, len(df), max_rows_per_chunk):
            end_row = min(start_row + max_rows_per_chunk, len(df))
            chunk_df = df.iloc[start_row:end_row]
            
            # Convert chunk to readable text
            chunk_text = self._dataframe_to_text(chunk_df)
            
            row_chunk = self._create_table_chunk(
                content=chunk_text,
                page_num=page_num,
                table_id=table_id,
                chunk_type="rows",
                summary=f"Rows {start_row + 1}-{end_row} of {table_id}",
                chunk_id=len(chunks)
            )
            chunks.append(row_chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean and filter empty sentences
        sentences = [clean_text(s) for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_text(self, current_chunk: List[str]) -> str:
        """
        Get overlap text from current chunk for next chunk.
        
        Args:
            current_chunk (List[str]): Current chunk sentences
            
        Returns:
            str: Overlap text
        """
        if not current_chunk:
            return ""
        
        # Get last sentence or part of it for overlap
        last_sentence = current_chunk[-1]
        words = last_sentence.split()
        
        if len(words) <= self.overlap_size:
            return last_sentence
        else:
            return ' '.join(words[-self.overlap_size:])
    
    def _create_text_chunk(self, text: str, page_num: int, section: str, chunk_id: int) -> Dict[str, Any]:
        """
        Create a text chunk with metadata.
        
        Args:
            text (str): Chunk text content
            page_num (int): Source page number
            section (str): Source section
            chunk_id (int): Chunk identifier
            
        Returns:
            Dict[str, Any]: Text chunk with metadata
        """
        return {
            "content": text,
            "content_type": "text",
            "page": page_num,
            "section": section,
            "chunk_id": f"text_{page_num}_{chunk_id}",
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    def _create_table_chunk(self, content: str, page_num: int, table_id: str,
                           chunk_type: str, summary: str, chunk_id: int) -> Dict[str, Any]:
        """
        Create a table chunk with metadata.
        
        Args:
            content (str): Chunk content
            page_num (int): Source page number
            table_id (str): Table identifier
            chunk_type (str): Type of table chunk (header, summary, rows)
            summary (str): Table summary
            chunk_id (int): Chunk identifier
            
        Returns:
            Dict[str, Any]: Table chunk with metadata
        """
        return {
            "content": content,
            "content_type": "table",
            "page": page_num,
            "table_id": table_id,
            "chunk_type": chunk_type,
            "summary": summary,
            "chunk_id": f"{table_id}_{chunk_type}_{chunk_id}",
            "word_count": len(content.split()),
            "char_count": len(content)
        }
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame chunk to readable text format.
        
        Args:
            df (pd.DataFrame): DataFrame chunk
            
        Returns:
            str: Readable text representation
        """
        if df.empty:
            return "Empty table section"
        
        text_parts = []
        
        # Add column headers
        headers = f"Columns: {', '.join(df.columns)}"
        text_parts.append(headers)
        
        # Add row data
        for idx, row in df.iterrows():
            row_text = []
            for col, value in row.items():
                if str(value).strip():  # Only include non-empty values
                    row_text.append(f"{col}: {value}")
            
            if row_text:
                text_parts.append(f"Row {idx + 1}: {'; '.join(row_text)}")
        
        return '. '.join(text_parts)
