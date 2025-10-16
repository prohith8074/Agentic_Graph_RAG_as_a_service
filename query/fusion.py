"""
Hybrid relevance scoring and result fusion.
Implements Reciprocal Rank Fusion (RRF) and other strategies for combining
results from multiple retrieval sources.
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class HybridScorer:
    """
    Handles the fusion of results from multiple retrieval methods using various scoring strategies.
    """

    def reciprocal_rank_fusion(self, results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Perform Reciprocal Rank Fusion on a list of ranked result lists.

        Args:
            results_list: A list where each item is a ranked list of documents (dicts).
                          Each document dict must have a unique identifier, e.g., 'id' or 'text'.
            k: A constant used in the RRF formula, defaults to 60.

        Returns:
            A single list of documents, reranked and scored according to RRF.
        """
        if not results_list:
            return []

        scores = defaultdict(float)
        doc_map = {}

        for results in results_list:
            for rank, result in enumerate(results, 1):
                # Use the document's text content as a unique identifier if no 'id' is present
                doc_id = result.get('id', result.get('text'))
                if not doc_id:
                    continue
                
                if doc_id not in doc_map:
                    doc_map[doc_id] = result
                
                # RRF formula: score(d) = sum(1 / (k + rank_i(d)))
                scores[doc_id] += 1.0 / (k + rank)

        # Sort documents by their fused score in descending order
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Create the final reranked list
        reranked_results = []
        for doc_id in sorted_doc_ids:
            final_doc = doc_map[doc_id]
            final_doc['fused_score'] = scores[doc_id]
            reranked_results.append(final_doc)

        logger.info(f"Fused {len(results_list)} result lists into {len(reranked_results)} documents using RRF.")
        return reranked_results
