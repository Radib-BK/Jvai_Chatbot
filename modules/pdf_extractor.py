"""
PDF Extractor Module
Handles extraction of text and tables from PDF documents with metadata tracking.
"""

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import re
from typing import List, Dict, Any
import logging
from .utils import clean_text, normalize_bullets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extract text and tables from PDF documents with metadata tracking.
    
    This class handles the extraction of both text content and structured tables
    from PDF files, maintaining metadata such as page numbers and section information.
    """
    
    def __init__(self):
        """Initialize the PDF extractor."""
        self.pages_text: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
    
    def extract_text_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text content using PyMuPDF with page and section metadata.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict[str, Any]]: List of text chunks with metadata
        """
        pages_text = []
        
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Opened PDF with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only process pages with content
                    # Clean and normalize the text
                    cleaned_text = clean_text(text)
                    normalized_text = normalize_bullets(cleaned_text)
                    
                    # Extract section headers (simple heuristic)
                    section = self._extract_section_header(normalized_text)
                    
                    page_data = {
                        "page": page_num + 1,
                        "section": section,
                        "text": normalized_text,
                        "char_count": len(normalized_text),
                        "word_count": len(normalized_text.split())
                    }
                    pages_text.append(page_data)
            
            doc.close()
            logger.info(f"Extracted text from {len(pages_text)} pages")
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
            raise
        
        return pages_text
    
    def extract_tables_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber with multiple strategies and page metadata.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict[str, Any]]: List of table data with metadata
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing {len(pdf.pages)} pages for table extraction")
                
                for page_num, page in enumerate(pdf.pages):
                    # Strategy 1: Default table extraction
                    page_tables = page.extract_tables()
                    
                    # Filter out malformed tables (tables that are just text layout)
                    valid_tables = []
                    if page_tables:
                        for table in page_tables:
                            if self._is_valid_table(table):
                                valid_tables.append(table)
                    
                    # Strategy 2: Text-based table extraction for financial documents
                    text_tables = self._extract_financial_tables_from_text(page)
                    
                    # Process valid extracted tables
                    for table_idx, table_data in enumerate(valid_tables):
                        if table_data and len(table_data) >= 1:
                            processed_table = self._process_complex_table(table_data)
                            
                            if processed_table is not None and not processed_table.empty:
                                table_info = {
                                    "page": page_num + 1,
                                    "table_id": f"table_{page_num + 1}_{table_idx + 1}",
                                    "df": processed_table,
                                    "rows": len(processed_table),
                                    "columns": len(processed_table.columns),
                                    "summary": self._generate_table_summary(processed_table)
                                }
                                tables.append(table_info)
                    
                    # Process text-based tables
                    for table_idx, text_table in enumerate(text_tables):
                        if text_table is not None and not text_table.empty:
                            table_info = {
                                "page": page_num + 1,
                                "table_id": f"text_table_{page_num + 1}_{table_idx + 1}",
                                "df": text_table,
                                "rows": len(text_table),
                                "columns": len(text_table.columns),
                                "summary": self._generate_table_summary(text_table)
                            }
                            tables.append(table_info)
                
                logger.info(f"Extracted {len(tables)} tables")
                
        except Exception as e:
            logger.error(f"Error extracting tables with pdfplumber: {str(e)}")
            raise
        
        return tables
    
    def extract_full_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract both text and tables from a PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Complete document data with text and tables
        """
        logger.info(f"Starting full document extraction for: {pdf_path}")
        
        # Extract text
        pages_text = self.extract_text_with_pymupdf(pdf_path)
        
        # Extract tables
        tables = self.extract_tables_with_pdfplumber(pdf_path)
        
        # Store in instance variables
        self.pages_text = pages_text
        self.tables = tables
        
        document_data = {
            "pages_text": pages_text,
            "tables": tables,
            "total_pages": len(pages_text),
            "total_tables": len(tables),
            "total_words": sum(page["word_count"] for page in pages_text),
            "source_file": pdf_path
        }
        
        logger.info(f"Extraction complete: {document_data['total_pages']} pages, "
                   f"{document_data['total_tables']} tables, "
                   f"{document_data['total_words']} words")
        
        return document_data
    
    def _extract_section_header(self, text: str) -> str:
        """
        Extract section header from text using simple heuristics.
        
        Args:
            text (str): Text content
            
        Returns:
            str: Detected section header or "General"
        """
        lines = text.split('\n')
        
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and (
                line.isupper() or 
                re.match(r'^\d+\.', line) or 
                re.match(r'^[A-Z][A-Za-z\s]+:?$', line)
            ):
                return line[:100]  # Limit header length
        
        return "General"
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize DataFrame data.
        
        Args:
            df (pd.DataFrame): Raw DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill NaN values with empty strings
        df = df.fillna('')
        
        # Clean column names
        df.columns = [clean_text(str(col)) for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)
        
        # Remove rows where all values are empty after cleaning
        df = df[~(df == '').all(axis=1)]
        
        return df
    
    def _generate_table_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a brief summary of table content.
        
        Args:
            df (pd.DataFrame): Table DataFrame
            
        Returns:
            str: Table summary
        """
        if df.empty:
            return "Empty table"
        
        summary_parts = [
            f"Table with {len(df)} rows and {len(df.columns)} columns",
            f"Columns: {', '.join(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}"
        ]
        
        # Add sample of first row if available
        if len(df) > 0:
            first_row_sample = ', '.join([str(val)[:20] for val in df.iloc[0].values[:2]])
            summary_parts.append(f"Sample data: {first_row_sample}")
        
        return "; ".join(summary_parts)

    def _process_complex_table(self, table_data: List[List]) -> pd.DataFrame:
        """
        Process complex table structures with merged cells and multi-line content.
        
        Args:
            table_data: Raw table data from pdfplumber
            
        Returns:
            pd.DataFrame: Processed table
        """
        if not table_data:
            return pd.DataFrame()
        
        # Clean and split multi-line cells
        processed_data = []
        for row in table_data:
            processed_row = []
            for cell in row:
                if cell is None:
                    processed_row.append("")
                else:
                    # Split multi-line content and take the first meaningful part
                    cell_parts = str(cell).split('\n')
                    meaningful_parts = [part.strip() for part in cell_parts if part.strip()]
                    processed_row.append(meaningful_parts[0] if meaningful_parts else "")
            processed_data.append(processed_row)
        
        # Determine headers and data
        if len(processed_data) >= 2:
            # Use first row as headers
            headers = processed_data[0]
            data_rows = processed_data[1:]
        else:
            # Single row - create generic headers
            headers = [f"Column_{i+1}" for i in range(len(processed_data[0]))]
            data_rows = processed_data
        
        # Create DataFrame
        try:
            df = pd.DataFrame(data_rows, columns=headers)
            return self._clean_dataframe(df)
        except Exception as e:
            logger.warning(f"Error creating DataFrame from complex table: {e}")
            return pd.DataFrame()
    
    def _extract_text_based_tables(self, page) -> List[pd.DataFrame]:
        """
        Extract tables from text patterns when standard table extraction fails.
        Specifically designed for financial policy documents with complex layouts.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List[pd.DataFrame]: List of extracted tables
        """
        tables = []
        
        try:
            # Get page text
            text = page.extract_text()
            if not text:
                return tables
            
            lines = text.split('\n')
            
            # Strategy 1: Look for explicit table markers
            table_start_patterns = [
                r'Table\s+\d+\.\d+\.\d+',  # Table 1.2.1 format
                r'Table\s+\d+\.\d+',       # Table 1.2 format
            ]
            
            current_table_lines = []
            in_table = False
            table_title = ""
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Check if this line starts a table
                for pattern in table_start_patterns:
                    if re.search(pattern, line):
                        # Save previous table if exists
                        if current_table_lines:
                            table_df = self._parse_text_table(current_table_lines, table_title)
                            if table_df is not None and not table_df.empty:
                                tables.append(table_df)
                        
                        # Start new table
                        current_table_lines = []
                        in_table = True
                        table_title = line
                        
                        # Look ahead for table content
                        for j in range(i+1, min(i+20, len(lines))):
                            next_line = lines[j].strip()
                            if next_line:
                                current_table_lines.append(next_line)
                        break
                
                # If we're in a table, collect structured lines
                if in_table and not any(re.search(p, line) for p in table_start_patterns):
                    # Look for lines that seem like table rows
                    words = line.split()
                    if len(words) >= 2:
                        # Check if line has structured data (numbers, financial terms, etc.)
                        has_numbers = any(re.search(r'[\d,.$%]+', word) for word in words)
                        has_financial_terms = any(word.lower() in ['budget', 'estimate', 'actual', '$m', 'million', '%'] for word in words)
                        
                        if has_numbers or has_financial_terms or len(words) >= 4:
                            current_table_lines.append(line)
                    elif not line:  # Empty line might end table
                        if len(current_table_lines) >= 2:  # Only if we have enough content
                            table_df = self._parse_text_table(current_table_lines, table_title)
                            if table_df is not None and not table_df.empty:
                                tables.append(table_df)
                        current_table_lines = []
                        in_table = False
            
            # Process final table if exists
            if current_table_lines:
                table_df = self._parse_text_table(current_table_lines, table_title)
                if table_df is not None and not table_df.empty:
                    tables.append(table_df)
            
            # Strategy 2: Look for column headers followed by data
            self._extract_header_based_tables(lines, tables)
        
        except Exception as e:
            logger.warning(f"Error in text-based table extraction: {e}")
        
        return tables
    
    def _parse_text_table(self, table_lines: List[str], title: str) -> pd.DataFrame:
        """Parse extracted table lines into a DataFrame."""
        if len(table_lines) < 2:
            return pd.DataFrame()
        
        try:
            # Try to identify headers and data
            parsed_rows = []
            
            for line in table_lines:
                # Split on multiple spaces or tabs to identify columns
                # This handles cases where data is separated by multiple spaces
                words = re.split(r'\s{2,}|\t+', line.strip())
                words = [w.strip() for w in words if w.strip()]
                
                if len(words) >= 2:
                    parsed_rows.append(words)
            
            if len(parsed_rows) >= 2:
                # Use first row as headers, or create generic headers
                headers = parsed_rows[0] if len(parsed_rows[0]) <= 6 else [f"Column_{i+1}" for i in range(len(parsed_rows[0]))]
                
                # Normalize all rows to same length
                max_cols = max(len(row) for row in parsed_rows)
                normalized_rows = []
                
                for row in parsed_rows[1:]:  # Skip header row
                    normalized_row = row + [''] * (max_cols - len(row))
                    normalized_rows.append(normalized_row[:max_cols])
                
                if normalized_rows:
                    # Ensure headers match data columns
                    headers = headers[:max_cols] + [f"Column_{i+1}" for i in range(len(headers), max_cols)]
                    
                    df = pd.DataFrame(normalized_rows, columns=headers[:max_cols])
                    df = self._clean_dataframe(df)
                    
                    return df
        
        except Exception as e:
            logger.warning(f"Error parsing text table: {e}")
        
        return pd.DataFrame()
    
    def _extract_header_based_tables(self, lines: List[str], tables: List[pd.DataFrame]):
        """Extract tables by looking for header patterns."""
        header_patterns = [
            r'Short\s+Term.*Long\s+Term',  # Financial objectives table
            r'Actual.*Budget.*Estimate',   # Budget tables
            r'\d{4}-\d{2}.*\d{4}-\d{2}',  # Year ranges
        ]
        
        for i, line in enumerate(lines):
            for pattern in header_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Found a potential header, look for data rows below
                    data_rows = []
                    
                    for j in range(i+1, min(i+15, len(lines))):
                        data_line = lines[j].strip()
                        if not data_line:
                            continue
                            
                        # Check if this looks like a data row
                        if re.search(r'[\d,.$%]+', data_line):
                            data_rows.append(data_line)
                        elif len(data_rows) > 0:
                            # End of table data
                            break
                    
                    if len(data_rows) >= 2:
                        # Try to create a table from this data
                        table_df = self._parse_text_table([line] + data_rows, f"Header-based table from line {i+1}")
                        if table_df is not None and not table_df.empty:
                            tables.append(table_df)
                    break

    def _is_valid_table(self, table_data: List[List]) -> bool:
        """Check if extracted table data represents a real table vs text layout."""
        if not table_data or len(table_data) < 2:
            return False
        
        first_row = table_data[0] if table_data else []
        none_count = sum(1 for cell in first_row if cell is None)
        if none_count > len(first_row) / 2:
            return False
        
        # Check for structured data
        has_numeric_data = False
        for row in table_data[:3]:
            for cell in row:
                if cell and re.search(r'[\d,.$%]+', str(cell)):
                    has_numeric_data = True
                    break
        
        return has_numeric_data
    
    def _extract_financial_tables_from_text(self, page) -> List[pd.DataFrame]:
        """Extract financial tables by parsing text content."""
        tables = []
        
        try:
            text = page.extract_text()
            if not text:
                return tables
            
            lines = text.split('\n')
            
            # Find table markers
            table_markers = []
            for i, line in enumerate(lines):
                if re.search(r'Table\s+\d+\.\d+\.\d+', line):
                    title = line.strip()
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not re.search(r'Table\s+\d+\.\d+\.\d+', next_line):
                            title += " - " + next_line
                    table_markers.append((i, title))
            
            # Extract each table
            for j, (line_num, title) in enumerate(table_markers):
                start_line = line_num + 1
                end_line = len(lines)
                
                if j + 1 < len(table_markers):
                    end_line = table_markers[j + 1][0]
                
                content_lines = []
                for i in range(start_line, min(start_line + 25, end_line)):
                    if i < len(lines):
                        line = lines[i].strip()
                        if line and not line.startswith('2005-06 Budget Paper'):
                            content_lines.append(line)
                
                parsed_table = self._parse_financial_table(content_lines, title)
                if parsed_table is not None and not parsed_table.empty:
                    tables.append(parsed_table)
        
        except Exception as e:
            logger.warning(f"Error in financial table extraction: {e}")
        
        return tables
    
    def _parse_financial_table(self, lines: List[str], title: str) -> pd.DataFrame:
        """Parse financial table based on type."""
        if not lines:
            return pd.DataFrame()
        
        try:
            if 'Financial Objectives' in title:
                return self._parse_objectives_table(lines)
            elif 'Budget' in title or 'Forward Estimates' in title:
                return self._parse_budget_table(lines)
            else:
                return self._parse_generic_financial_table(lines)
        except Exception as e:
            logger.warning(f"Error parsing table '{title}': {e}")
            return pd.DataFrame()
    
    def _parse_objectives_table(self, lines: List[str]) -> pd.DataFrame:
        """Parse Financial Objectives table."""
        short_term = []
        long_term = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'Short Term' in line and 'Financial Objectives' in line:
                current_section = 'short'
                continue
            elif 'Long Term' in line and 'Financial Objectives' in line:
                current_section = 'long'
                continue
            elif 'Key Measures' in line:
                break
            
            if current_section == 'short' and line and len(line) > 10:
                short_term.append(line)
            elif current_section == 'long' and line and len(line) > 10:
                long_term.append(line)
        
        if short_term or long_term:
            max_len = max(len(short_term), len(long_term))
            short_term.extend([''] * (max_len - len(short_term)))
            long_term.extend([''] * (max_len - len(long_term)))
            
            return pd.DataFrame({
                'Short Term Financial Objectives': short_term,
                'Long Term Financial Objectives': long_term
            })
        
        return pd.DataFrame()
    
    def _parse_budget_table(self, lines: List[str]) -> pd.DataFrame:
        """Parse budget table."""
        data_rows = []
        
        for line in lines:
            if re.search(r'[\d,]+\.?\d*', line) and len(line.split()) >= 3:
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 2:
                    data_rows.append(parts)
        
        if len(data_rows) >= 2:
            max_cols = max(len(row) for row in data_rows)
            headers = [f'Column_{i+1}' for i in range(max_cols)]
            return pd.DataFrame(data_rows, columns=headers[:max_cols])
        
        return pd.DataFrame()
    
    def _parse_generic_financial_table(self, lines: List[str]) -> pd.DataFrame:
        """Parse generic financial table."""
        return self._parse_budget_table(lines)
