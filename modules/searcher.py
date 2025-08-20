"""
Searcher Module
Handles vector search and result ranking for query processing.
"""

from typing import List, Dict, Any, Set
import logging
from .embedding_index import EmbeddingIndex
from .utils import format_citation, extract_keywords, log_operation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentSearcher:
    """
    Handles semantic search and result ranking for document queries.
    
    This class provides high-level search functionality including
    result deduplication, ranking, and citation formatting.
    """
    
    def __init__(self, embedding_index: EmbeddingIndex):
        """
        Initialize the content searcher.
        
        Args:
            embedding_index (EmbeddingIndex): Initialized embedding index
        """
        self.embedding_index = embedding_index
        self.search_history: List[Dict[str, Any]] = []
    
    def search(self, query: str, top_k: int = 10, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform semantic search with result filtering and ranking.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to retrieve
            min_score (float): Minimum similarity score threshold
            
        Returns:
            List[Dict[str, Any]]: Ranked and filtered search results
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k}, min_score={min_score})")
        
        try:
            # Get raw search results from embedding index
            raw_results = self.embedding_index.search(query, top_k * 2)  # Get extra for filtering
            
            if not raw_results:
                logger.info("No search results found")
                return []
            
            # Filter by minimum score
            filtered_results = [
                result for result in raw_results 
                if result["score"] >= min_score
            ]
            
            # Deduplicate results
            deduplicated_results = self._deduplicate_results(filtered_results)
            
            # Rank and enhance results
            ranked_results = self._rank_results(deduplicated_results, query)
            
            # Limit to requested number
            final_results = ranked_results[:top_k]
            
            # Add search to history
            self._add_to_history(query, final_results)
            
            logger.info(f"Search completed. Returning {len(final_results)} results")
            log_operation("Search", True, f"Query: '{query}', Results: {len(final_results)}")
            
            return final_results
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            log_operation("Search", False, error_msg)
            return []
    
    def search_with_context(self, query: str, conversation_context: List[str] = None, 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with conversation context for better results.
        
        Args:
            query (str): Search query
            conversation_context (List[str], optional): Previous conversation turns
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Context-aware search results
        """
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(query, conversation_context)
        
        logger.info(f"Context-enhanced search: '{enhanced_query}'")
        
        # Perform regular search with enhanced query
        results = self.search(enhanced_query, top_k)
        
        # Add context relevance scoring
        for result in results:
            result["context_relevance"] = self._calculate_context_relevance(
                result, query, conversation_context
            )
        
        # Re-sort by context relevance
        results.sort(key=lambda x: x["context_relevance"], reverse=True)
        
        return results
    
    def get_diverse_results(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get diverse search results covering different content types and sources.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Diverse search results
        """
        # Get more results initially
        all_results = self.search(query, top_k * 3)
        
        if not all_results:
            return []
        
        # Separate by content type
        text_results = [r for r in all_results if r["metadata"].get("content_type") == "text"]
        table_results = [r for r in all_results if r["metadata"].get("content_type") == "table"]
        
        # Ensure diversity in results
        diverse_results = []
        
        # Add best text results
        diverse_results.extend(text_results[:max(1, top_k // 2)])
        
        # Add best table results
        diverse_results.extend(table_results[:max(1, top_k // 2)])
        
        # Fill remaining slots with top results
        remaining_slots = top_k - len(diverse_results)
        for result in all_results:
            if len(diverse_results) >= top_k:
                break
            if result not in diverse_results:
                diverse_results.append(result)
        
        return diverse_results[:top_k]
    
    def search_tables_only(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search specifically for table content.
        
        Args:
            query (str): Search query
            top_k (int): Number of table results to return
            
        Returns:
            List[Dict[str, Any]]: Table search results
        """
        all_results = self.search(query, top_k * 2)
        
        # Filter for table content only
        table_results = [
            result for result in all_results 
            if result["metadata"].get("content_type") == "table"
        ]
        
        return table_results[:top_k]
    
    def search_text_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically for text content.
        
        Args:
            query (str): Search query
            top_k (int): Number of text results to return
            
        Returns:
            List[Dict[str, Any]]: Text search results
        """
        all_results = self.search(query, top_k * 2)
        
        # Filter for text content only
        text_results = [
            result for result in all_results 
            if result["metadata"].get("content_type") == "text"
        ]
        
        return text_results[:top_k]
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """
        Get search history for conversation memory.
        
        Returns:
            List[Dict[str, Any]]: Search history
        """
        return self.search_history.copy()
    
    def clear_search_history(self) -> None:
        """Clear search history."""
        self.search_history.clear()
        logger.info("Search history cleared")
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate or highly similar results.
        
        Args:
            results (List[Dict[str, Any]]): Raw search results
            
        Returns:
            List[Dict[str, Any]]: Deduplicated results
        """
        if not results:
            return results
        
        deduplicated = []
        seen_pages: Set[int] = set()
        seen_content_hashes: Set[str] = set()
        
        for result in results:
            metadata = result["metadata"]
            page = metadata.get("page", 0)
            content = result["content"]
            
            # Simple content hash for duplicate detection
            content_hash = hash(content[:100])  # Hash first 100 characters
            
            # Skip if we've seen very similar content from the same page
            if page in seen_pages and content_hash in seen_content_hashes:
                continue
            
            # Skip exact content duplicates
            if content_hash in seen_content_hashes:
                # But allow if it's from a different page and significantly different score
                if page not in seen_pages:
                    pass  # Allow different page
                else:
                    continue
            
            deduplicated.append(result)
            seen_pages.add(page)
            seen_content_hashes.add(content_hash)
        
        logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
        return deduplicated
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-rank results based on multiple factors.
        
        Args:
            results (List[Dict[str, Any]]): Search results
            query (str): Original query
            
        Returns:
            List[Dict[str, Any]]: Re-ranked results
        """
        if not results:
            return results
        
        # Extract query keywords for relevance scoring
        query_keywords = extract_keywords(query.lower())
        
        for result in results:
            # Start with similarity score
            base_score = result["score"]
            
            # Keyword matching bonus
            content_lower = result["content"].lower()
            keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
            keyword_bonus = (keyword_matches / len(query_keywords)) * 0.1 if query_keywords else 0
            
            # Content type preference (slight preference for text over tables for general queries)
            content_type_bonus = 0.05 if result["metadata"].get("content_type") == "text" else 0
            
            # Length penalty for very short content
            content_length = len(result["content"])
            length_penalty = 0 if content_length > 50 else -0.05
            
            # Calculate final ranking score
            final_score = base_score + keyword_bonus + content_type_bonus + length_penalty
            result["ranking_score"] = final_score
            
            # Add detailed scoring breakdown
            result["score_breakdown"] = {
                "similarity": base_score,
                "keyword_bonus": keyword_bonus,
                "content_type_bonus": content_type_bonus,
                "length_penalty": length_penalty,
                "final": final_score
            }
        
        # Sort by final ranking score
        results.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        return results
    
    def _enhance_query_with_context(self, query: str, context: List[str] = None) -> str:
        """
        Enhance query with conversation context.
        
        Args:
            query (str): Original query
            context (List[str], optional): Conversation context
            
        Returns:
            str: Enhanced query
        """
        if not context:
            return query
        
        # Extract keywords from recent context
        context_text = " ".join(context[-3:])  # Use last 3 context items
        context_keywords = extract_keywords(context_text)
        
        # Add relevant context keywords to query
        enhanced_parts = [query]
        for keyword in context_keywords[:3]:  # Limit to 3 context keywords
            if keyword.lower() not in query.lower():
                enhanced_parts.append(keyword)
        
        return " ".join(enhanced_parts)
    
    def _calculate_context_relevance(self, result: Dict[str, Any], 
                                   query: str, context: List[str] = None) -> float:
        """
        Calculate relevance based on conversation context.
        
        Args:
            result (Dict[str, Any]): Search result
            query (str): Original query
            context (List[str], optional): Conversation context
            
        Returns:
            float: Context relevance score
        """
        if not context:
            return result["score"]
        
        base_score = result["score"]
        content = result["content"].lower()
        
        # Check for context keyword matches
        context_text = " ".join(context).lower()
        context_keywords = extract_keywords(context_text)
        
        context_matches = sum(1 for keyword in context_keywords if keyword in content)
        context_bonus = (context_matches / len(context_keywords)) * 0.15 if context_keywords else 0
        
        return base_score + context_bonus
    
    def _add_to_history(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Add search to history for conversation memory.
        
        Args:
            query (str): Search query
            results (List[Dict[str, Any]]): Search results
        """
        history_entry = {
            "query": query,
            "results_count": len(results),
            "top_score": results[0]["score"] if results else 0,
            "content_types": list(set(r["metadata"].get("content_type", "unknown") for r in results))
        }
        
        self.search_history.append(history_entry)
        
        # Keep only recent history (last 10 searches)
        if len(self.search_history) > 10:
            self.search_history = self.search_history[-10:]
