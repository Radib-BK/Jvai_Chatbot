"""
Embedding Index Module
Creates and manages FAISS vector index for semantic search.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import logging
import pickle
import os
from .utils import log_operation, create_progress_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """
    Creates and manages FAISS vector index for semantic search of document content.
    
    This class handles the creation of embeddings using sentence transformers
    and builds a FAISS index for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding index with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            log_operation("Model Loading", True, f"Dimension: {self.embedding_dim}")
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            logger.error(error_msg)
            log_operation("Model Loading", False, error_msg)
            raise
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks (List[Dict[str, Any]]): List of text chunks with metadata
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not chunks:
            return np.array([])
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        
        # Extract text content from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        try:
            # Create embeddings in batches for memory efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                progress_msg = create_progress_message(i + len(batch_texts), len(texts), "Embedding")
                logger.info(progress_msg)
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            log_operation("Embedding Creation", True, f"Shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to create embeddings: {str(e)}"
            logger.error(error_msg)
            log_operation("Embedding Creation", False, error_msg)
            raise
    
    def build_index(self, text_chunks: List[Dict[str, Any]], 
                   table_chunks: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from text and table chunks.
        
        Args:
            text_chunks (List[Dict[str, Any]]): Text chunks with metadata
            table_chunks (List[Dict[str, Any]]): Table chunks with metadata
        """
        logger.info("Building FAISS index")
        
        # Combine all chunks
        all_chunks = text_chunks + table_chunks
        
        if not all_chunks:
            logger.warning("No chunks provided for indexing")
            return
        
        try:
            # Create embeddings
            embeddings = self.create_embeddings(all_chunks)
            
            if embeddings.size == 0:
                logger.warning("No embeddings created")
                return
            
            # Store metadata
            self.metadata = all_chunks
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
            log_operation("Index Building", True, f"Vectors: {self.index.ntotal}")
            
        except Exception as e:
            error_msg = f"Failed to build index: {str(e)}"
            logger.error(error_msg)
            log_operation("Index Building", False, error_msg)
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for similar content.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: Search results with scores and metadata
        """
        if not self.index or not self.metadata:
            logger.warning("Index not built or no metadata available")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Search the index
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):  # Valid index
                    result = {
                        "content": self.metadata[idx]["content"],
                        "metadata": self.metadata[idx],
                        "score": float(score),
                        "relevance": self._calculate_relevance(float(score))
                    }
                    results.append(result)
            
            logger.info(f"Search completed. Found {len(results)} results for query: '{query[:50]}...'")
            
            return results
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            log_operation("Search", False, error_msg)
            return []
    
    def get_embeddings_for_query(self, query: str) -> np.ndarray:
        """
        Get embeddings for a query string.
        
        Args:
            query (str): Query string
            
        Returns:
            np.ndarray: Query embeddings
        """
        try:
            return self.model.encode([query], normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Failed to create query embeddings: {str(e)}")
            return np.array([])
    
    def save_index(self, file_path: str) -> bool:
        """
        Save the FAISS index and metadata to file.
        
        Args:
            file_path (str): Path to save the index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.index:
                logger.warning("No index to save")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{file_path}.faiss")
            
            # Save metadata
            with open(f"{file_path}.metadata", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            logger.info(f"Index saved to {file_path}")
            log_operation("Index Saving", True, file_path)
            return True
            
        except Exception as e:
            error_msg = f"Failed to save index: {str(e)}"
            logger.error(error_msg)
            log_operation("Index Saving", False, error_msg)
            return False
    
    def load_index(self, file_path: str) -> bool:
        """
        Load FAISS index and metadata from file.
        
        Args:
            file_path (str): Path to load the index from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load FAISS index
            if not os.path.exists(f"{file_path}.faiss"):
                logger.warning(f"Index file not found: {file_path}.faiss")
                return False
            
            self.index = faiss.read_index(f"{file_path}.faiss")
            
            # Load metadata
            with open(f"{file_path}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                saved_model_name = data.get('model_name', self.model_name)
                self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
            
            # Verify model compatibility
            if saved_model_name != self.model_name:
                logger.warning(f"Model mismatch: saved={saved_model_name}, current={self.model_name}")
            
            logger.info(f"Index loaded from {file_path} with {self.index.ntotal} vectors")
            log_operation("Index Loading", True, f"Vectors: {self.index.ntotal}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load index: {str(e)}"
            logger.error(error_msg)
            log_operation("Index Loading", False, error_msg)
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dict[str, Any]: Index statistics
        """
        if not self.index:
            return {"status": "No index built"}
        
        stats = {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "metadata_count": len(self.metadata),
            "text_chunks": len([m for m in self.metadata if m.get("content_type") == "text"]),
            "table_chunks": len([m for m in self.metadata if m.get("content_type") == "table"])
        }
        
        return stats
    
    def _calculate_relevance(self, score: float) -> str:
        """
        Calculate relevance category based on similarity score.
        
        Args:
            score (float): Similarity score
            
        Returns:
            str: Relevance category
        """
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
