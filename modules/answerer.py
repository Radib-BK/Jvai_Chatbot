"""
Answerer Module
Generates comprehensive answers with citations and conversation memory.
"""

from typing import List, Dict, Any, Optional
import logging
from .searcher import ContentSearcher
from .utils import format_citation, format_table_for_display, truncate_text, log_operation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationalAnswerer:
    """
    Generates contextual answers with citations and maintains conversation memory.
    
    This class creates comprehensive answers by combining search results,
    formatting citations, and maintaining conversation context.
    """
    
    def __init__(self, searcher: ContentSearcher):
        """
        Initialize the conversational answerer.
        
        Args:
            searcher (ContentSearcher): Content searcher instance
        """
        self.searcher = searcher
        self.conversation_memory: List[Dict[str, Any]] = []
        self.max_memory_turns = 10
    
    def answer_question(self, question: str, use_context: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive answer to a question.
        
        Args:
            question (str): User question
            use_context (bool): Whether to use conversation context
            
        Returns:
            Dict[str, Any]: Complete answer with citations and metadata
        """
        logger.info(f"Answering question: '{question}'")
        
        try:
            # Get conversation context if requested
            context = self._get_conversation_context() if use_context else None
            
            # Search for relevant content
            if use_context and context:
                search_results = self.searcher.search_with_context(
                    question, context, top_k=5
                )
            else:
                search_results = self.searcher.get_diverse_results(question, top_k=5)
            
            if not search_results:
                return self._create_no_results_response(question)
            
            # Generate answer
            answer_data = self._generate_answer_from_results(question, search_results)
            
            # Add to conversation memory
            self._add_to_memory(question, answer_data)
            
            logger.info(f"Answer generated successfully for: '{question}'")
            log_operation("Answer Generation", True, f"Results used: {len(search_results)}")
            
            return answer_data
            
        except Exception as e:
            error_msg = f"Failed to answer question: {str(e)}"
            logger.error(error_msg)
            log_operation("Answer Generation", False, error_msg)
            return self._create_error_response(question, error_msg)
    
    def answer_with_tables_focus(self, question: str) -> Dict[str, Any]:
        """
        Generate answer with focus on table content.
        
        Args:
            question (str): User question
            
        Returns:
            Dict[str, Any]: Answer focused on table data
        """
        logger.info(f"Answering with table focus: '{question}'")
        
        # Search for table content specifically
        table_results = self.searcher.search_tables_only(question, top_k=3)
        text_results = self.searcher.search_text_only(question, top_k=2)
        
        # Combine results with table priority
        all_results = table_results + text_results
        
        if not all_results:
            return self._create_no_results_response(question)
        
        # Generate table-focused answer
        answer_data = self._generate_table_focused_answer(question, all_results)
        
        # Add to memory
        self._add_to_memory(question, answer_data)
        
        return answer_data
    
    def summarize_document_section(self, section_query: str) -> Dict[str, Any]:
        """
        Provide a summary of a document section.
        
        Args:
            section_query (str): Query about document section
            
        Returns:
            Dict[str, Any]: Section summary
        """
        logger.info(f"Summarizing section: '{section_query}'")
        
        # Get comprehensive results for summarization
        results = self.searcher.search(section_query, top_k=10, min_score=0.2)
        
        if not results:
            return self._create_no_results_response(section_query)
        
        # Generate summary
        summary_data = self._generate_summary_from_results(section_query, results)
        
        return summary_data
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history for memory display.
        
        Returns:
            List[Dict[str, Any]]: Conversation history
        """
        return self.conversation_memory.copy()
    
    def clear_conversation_memory(self) -> None:
        """Clear conversation memory."""
        self.conversation_memory.clear()
        self.searcher.clear_search_history()
        logger.info("Conversation memory cleared")
    
    def _generate_answer_from_results(self, question: str, 
                                    results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive answer from search results.
        
        Args:
            question (str): Original question
            results (List[Dict[str, Any]]): Search results
            
        Returns:
            Dict[str, Any]: Complete answer data
        """
        # Separate text and table results
        text_results = [r for r in results if r["metadata"].get("content_type") == "text"]
        table_results = [r for r in results if r["metadata"].get("content_type") == "table"]
        
        # Build answer components
        answer_parts = []
        citations = []
        confidence_scores = []
        
        # Process text results
        if text_results:
            text_answer = self._create_text_answer(question, text_results)
            if text_answer:
                answer_parts.append(text_answer)
                confidence_scores.extend([r["score"] for r in text_results[:3]])
        
        # Process table results
        if table_results:
            table_answer = self._create_table_answer(question, table_results)
            if table_answer:
                answer_parts.append(table_answer)
                confidence_scores.extend([r["score"] for r in table_results[:2]])
        
        # Combine answer parts
        if answer_parts:
            full_answer = "\n\n".join(answer_parts)
        else:
            full_answer = "I couldn't find specific information to answer your question."
        
        # Generate citations
        citations = self._generate_citations(results[:5])  # Top 5 results for citations
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "question": question,
            "answer": full_answer,
            "citations": citations,
            "confidence": avg_confidence,
            "sources_used": len(results),
            "has_tables": len(table_results) > 0,
            "has_text": len(text_results) > 0,
            "result_types": {
                "text_results": len(text_results),
                "table_results": len(table_results)
            }
        }
    
    def _create_text_answer(self, question: str, text_results: List[Dict[str, Any]]) -> str:
        """
        Create answer section from text results.
        
        Args:
            question (str): Original question
            text_results (List[Dict[str, Any]]): Text search results
            
        Returns:
            str: Text-based answer section
        """
        if not text_results:
            return ""
        
        # Use top text results to build answer
        answer_pieces = []
        
        for result in text_results[:3]:  # Use top 3 text results
            content = result["content"]
            metadata = result["metadata"]
            
            # Truncate content if too long
            if len(content) > 300:
                content = truncate_text(content, 300)
            
            # Add content with basic formatting
            page_ref = f"(Page {metadata.get('page', 'N/A')})"
            answer_pieces.append(f"{content} {page_ref}")
        
        return "Based on the document content:\n\n" + "\n\n".join(answer_pieces)
    
    def _create_table_answer(self, question: str, table_results: List[Dict[str, Any]]) -> str:
        """
        Create answer section from table results.
        
        Args:
            question (str): Original question
            table_results (List[Dict[str, Any]]): Table search results
            
        Returns:
            str: Table-based answer section
        """
        if not table_results:
            return ""
        
        answer_parts = ["Relevant table information:"]
        
        for result in table_results[:2]:  # Use top 2 table results
            metadata = result["metadata"]
            content = result["content"]
            
            table_id = metadata.get("table_id", "Unknown")
            page = metadata.get("page", "N/A")
            
            # Format table content
            if "df" in metadata:
                # If we have the actual DataFrame
                table_display = format_table_for_display(metadata["df"], max_rows=3)
                answer_parts.append(f"\nTable {table_id} (Page {page}):\n{table_display}")
            else:
                # Use the content text
                truncated_content = truncate_text(content, 200)
                answer_parts.append(f"\nTable {table_id} (Page {page}): {truncated_content}")
        
        return "\n".join(answer_parts)
    
    def _generate_table_focused_answer(self, question: str, 
                                     results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer with special focus on table content.
        
        Args:
            question (str): Original question
            results (List[Dict[str, Any]]): Search results
            
        Returns:
            Dict[str, Any]: Table-focused answer
        """
        table_results = [r for r in results if r["metadata"].get("content_type") == "table"]
        
        if not table_results:
            return self._create_no_results_response(question, 
                                                  "No relevant tables found for your question.")
        
        answer_parts = ["Here's the table information relevant to your question:"]
        citations = []
        
        for result in table_results:
            metadata = result["metadata"]
            table_id = metadata.get("table_id", "Unknown")
            page = metadata.get("page", "N/A")
            summary = metadata.get("summary", "")
            
            # Add table summary and details
            answer_parts.append(f"\n**Table {table_id}** (Page {page})")
            if summary:
                answer_parts.append(f"Summary: {summary}")
            
            # Add formatted table content
            if "df" in metadata:
                table_display = format_table_for_display(metadata["df"], max_rows=5)
                answer_parts.append(f"\n{table_display}")
            
            # Add citation
            citation = format_citation(page, table_id)
            citations.append(citation)
        
        return {
            "question": question,
            "answer": "\n".join(answer_parts),
            "citations": citations,
            "confidence": table_results[0]["score"] if table_results else 0,
            "sources_used": len(table_results),
            "has_tables": True,
            "has_text": False,
            "result_types": {
                "text_results": 0,
                "table_results": len(table_results)
            }
        }
    
    def _generate_summary_from_results(self, query: str, 
                                     results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary from multiple results.
        
        Args:
            query (str): Summary query
            results (List[Dict[str, Any]]): Search results
            
        Returns:
            Dict[str, Any]: Summary data
        """
        # Group results by page
        page_groups = {}
        for result in results:
            page = result["metadata"].get("page", 0)
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(result)
        
        summary_parts = [f"Summary for: {query}\n"]
        
        # Create summary by page
        for page in sorted(page_groups.keys()):
            page_results = page_groups[page]
            
            summary_parts.append(f"\n**Page {page}:**")
            
            # Combine content from this page
            page_content = []
            for result in page_results[:3]:  # Limit per page
                content = truncate_text(result["content"], 150)
                page_content.append(content)
            
            summary_parts.append(" ".join(page_content))
        
        # Generate citations
        citations = self._generate_citations(results[:10])
        
        return {
            "question": query,
            "answer": "\n".join(summary_parts),
            "citations": citations,
            "confidence": sum(r["score"] for r in results[:5]) / min(5, len(results)),
            "sources_used": len(results),
            "summary_type": "multi_page",
            "pages_covered": list(page_groups.keys())
        }
    
    def _generate_citations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Generate formatted citations from results.
        
        Args:
            results (List[Dict[str, Any]]): Search results
            
        Returns:
            List[str]: Formatted citations
        """
        citations = []
        seen_citations = set()
        
        for result in results:
            metadata = result["metadata"]
            page = metadata.get("page", 0)
            table_id = metadata.get("table_id")
            section = metadata.get("section")
            
            citation = format_citation(page, table_id, section)
            
            if citation not in seen_citations:
                citations.append(citation)
                seen_citations.add(citation)
        
        return citations
    
    def _get_conversation_context(self) -> List[str]:
        """
        Get recent conversation context.
        
        Returns:
            List[str]: Recent conversation context
        """
        context = []
        
        # Get recent questions and answer snippets
        for memory_item in self.conversation_memory[-3:]:
            context.append(memory_item["question"])
            
            # Add snippet of previous answer
            answer_snippet = truncate_text(memory_item["answer"], 100)
            context.append(answer_snippet)
        
        return context
    
    def _add_to_memory(self, question: str, answer_data: Dict[str, Any]) -> None:
        """
        Add Q&A to conversation memory.
        
        Args:
            question (str): User question
            answer_data (Dict[str, Any]): Answer data
        """
        memory_item = {
            "question": question,
            "answer": answer_data["answer"],
            "confidence": answer_data.get("confidence", 0),
            "sources_used": answer_data.get("sources_used", 0),
            "timestamp": len(self.conversation_memory)  # Simple timestamp
        }
        
        self.conversation_memory.append(memory_item)
        
        # Keep memory within limits
        if len(self.conversation_memory) > self.max_memory_turns:
            self.conversation_memory = self.conversation_memory[-self.max_memory_turns:]
    
    def _create_no_results_response(self, question: str, 
                                  custom_message: str = None) -> Dict[str, Any]:
        """
        Create response when no results are found.
        
        Args:
            question (str): Original question
            custom_message (str, optional): Custom message
            
        Returns:
            Dict[str, Any]: No results response
        """
        message = custom_message or "I couldn't find relevant information in the document to answer your question. Please try rephrasing your question or asking about different topics covered in the document."
        
        return {
            "question": question,
            "answer": message,
            "citations": [],
            "confidence": 0,
            "sources_used": 0,
            "has_tables": False,
            "has_text": False,
            "result_types": {
                "text_results": 0,
                "table_results": 0
            }
        }
    
    def _create_error_response(self, question: str, error_message: str) -> Dict[str, Any]:
        """
        Create response for error conditions.
        
        Args:
            question (str): Original question
            error_message (str): Error description
            
        Returns:
            Dict[str, Any]: Error response
        """
        return {
            "question": question,
            "answer": f"I encountered an error while processing your question: {error_message}",
            "citations": [],
            "confidence": 0,
            "sources_used": 0,
            "has_tables": False,
            "has_text": False,
            "error": True,
            "error_message": error_message
        }
