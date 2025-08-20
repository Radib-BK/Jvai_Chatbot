"""
Configuration file for the Financial Policy Chatbot.
Adjust these settings to customize the application behavior.
"""

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
EMBEDDING_DIMENSION = 384  # Dimension of the embedding vectors

# Chunking Configuration
MAX_CHUNK_SIZE = 500  # Maximum words per chunk
OVERLAP_SIZE = 50     # Word overlap between chunks

# Search Configuration
DEFAULT_TOP_K = 5        # Default number of search results
MIN_SCORE_THRESHOLD = 0.3  # Minimum similarity score
MAX_SEARCH_RESULTS = 20    # Maximum results to consider before ranking

# Answer Generation Configuration
MAX_MEMORY_TURNS = 10      # Maximum conversation history to maintain
MAX_ANSWER_LENGTH = 1000   # Maximum characters in generated answers
CITATION_FORMAT = "Source: Page {page}{table_info}{section_info}"

# Table Processing Configuration
MAX_TABLE_ROWS_DISPLAY = 5   # Maximum table rows to show in interface
MAX_TABLE_CELL_LENGTH = 20   # Maximum characters per cell in display

# File Processing Configuration
ALLOWED_FILE_TYPES = ['pdf']  # Supported file types
MAX_FILE_SIZE_MB = 50        # Maximum file size in MB
TEMP_DIR = "temp"            # Temporary file directory

# UI Configuration
PAGE_TITLE = "Financial Policy Chatbot"
PAGE_ICON = "💼"
LAYOUT = "wide"

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Configuration
BATCH_SIZE = 32              # Batch size for embedding generation
USE_GPU = False              # Whether to use GPU acceleration (if available)
CACHE_EMBEDDINGS = True      # Whether to cache embeddings for reuse

# Advanced Search Configuration
ENABLE_CONTEXT_SEARCH = True    # Enable conversation context in search
ENABLE_DIVERSE_RESULTS = True   # Ensure diversity in search results
KEYWORD_BOOST_FACTOR = 0.1      # Boost factor for keyword matches

# Table Extraction Configuration
TABLE_MIN_ROWS = 2              # Minimum rows to consider as table
TABLE_MIN_COLS = 2              # Minimum columns to consider as table
CLEAN_TABLE_HEADERS = True      # Whether to clean table headers

# Citation Configuration
SHOW_PAGE_NUMBERS = True        # Show page numbers in citations
SHOW_TABLE_IDS = True          # Show table IDs in citations
SHOW_SECTION_INFO = True       # Show section information in citations

# Error Handling Configuration
MAX_RETRY_ATTEMPTS = 3         # Maximum retry attempts for failed operations
TIMEOUT_SECONDS = 30           # Timeout for long operations

# Development Configuration
DEBUG_MODE = False             # Enable debug mode
VERBOSE_LOGGING = False        # Enable verbose logging
SAVE_INTERMEDIATE_RESULTS = False  # Save intermediate processing results
