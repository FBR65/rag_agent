"""
Licence and Copyright

The use of this code or parts of it is permitted exclusively for private,
educational, or non-commercial purposes.
Any commercial use or use by governmental organizations is prohibited without prior
written permission from the author.

Copyright 2025 Frank Reis
"""

import logging

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from typing import List, Dict, Optional
from openai import OpenAI
import numpy as np
from pydantic import BaseModel, Field

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoredPoint(BaseModel):
    id: str = Field(description="Unique identifier for the point")
    version: int = Field(description="Version of the point")
    score: float = Field(description="Score of the point")
    payload: Dict = Field(description="Payload data")
    vector: Optional[List[float]] = Field(
        None, description="Vector representation of the point"
    )
    shard_key: Optional[str] = Field(None, description="Shard key for the point")
    order_value: Optional[float] = Field(None, description="Order value for the point")


class Hybrid_search:
    """
    A class for performing hybrid search using dense and sparse embeddings, and full-text search.
    """

    def __init__(self, collection_name_prefix) -> None:
        """
        Initialize the Hybrid_search object with dense and sparse embedding models, a Qdrant client,
        and the collection name prefix.
        """
        self.collection_name_prefix = collection_name_prefix
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="my key",  # required but ignored
        )
        self.embedding_model = "paraphrase-multilingual:latest"
        self.reranker_model = "bge-m3:latest"
        self.sparse_embedding_model = SparseTextEmbedding(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
        )

        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT
        )

    def rerank_results(
        self, query: str, results: List[models.ScoredPoint]
    ) -> List[models.ScoredPoint]:
        """
        Reranks the given search results using a reranker model.

        Args:
            query (str): The query string.
            results (List[models.ScoredPoint]): A list of Qdrant ScoredPoint objects.

        Returns:
            List[models.ScoredPoint]: A list of Qdrant ScoredPoint objects, reranked based on relevance.
        """
        if not results:
            return []

        pairs = [(query, result.payload["text"]) for result in results]

        # Prepare input for the reranker model
        reranker_input = [f"query: {pair[0]} document: {pair[1]}" for pair in pairs]

        # Call the reranker model
        response = self.client.embeddings.create(
            model=self.reranker_model, input=reranker_input
        )

        # Extract the rerank scores from the response
        rerank_scores = [data.embedding[0] for data in response.data]

        # Sort results by the rerank scores
        reranked_results = sorted(
            zip(results, rerank_scores), key=lambda item: item[1], reverse=True
        )
        return [result for result, score in reranked_results]

    def query_hybrid_search(self, query, metadata_filter=None, limit=20):
        """
        Performs hybrid search across all distance collections and returns raw, reranked per collection,
        and reranked combined results.

        Args:
            query (str): The query string.
            metadata_filter (models.Filter, optional): A Qdrant filter object based on metadata. Defaults to None.
            limit (int, optional): The maximum number of results to return per collection and for the combined rerank.

        Returns:
            dict: A dictionary containing raw results per collection, reranked results per collection,
                    and reranked combined results.
        """
        response = self.client.embeddings.create(
            input=[query], model=self.embedding_model
        )
        dense_query = response.data[0].embedding
        sparse_query = list(self.sparse_embedding_model.embed([query]))[0]
        collection_distances = ["COSINE", "EUCLID", "DOT", "MANHATTAN"]
        all_raw_results = []
        raw_results_per_collection = {}
        reranked_results_per_collection = {}

        for distance_type in collection_distances:
            collection_name = f"{self.collection_name_prefix}_{distance_type}"
            logger.info(f"Performing hybrid search on collection: {collection_name}")
            try:
                raw_results = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_query.indices.tolist(),
                                values=sparse_query.values.tolist(),
                            ),
                            using="sparse",
                            limit=limit,
                        ),
                        models.Prefetch(
                            query=dense_query,
                            using="dense",
                            limit=limit,
                        ),
                    ],
                    query_filter=metadata_filter,
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF
                    ),  # Reciprocal Rerank Fusion
                    limit=limit,
                )
                raw_results_per_collection[collection_name] = raw_results.points
                all_raw_results.extend(raw_results.points)

                # Rerank results for the current collection
                reranked_collection_results = self.rerank_results(
                    query, raw_results.points
                )
                reranked_results_per_collection[collection_name] = (
                    reranked_collection_results
                )

            except Exception as e:
                logger.error(
                    f"Error during hybrid search on collection {collection_name}: {e}"
                )
                raw_results_per_collection[collection_name] = []
                reranked_results_per_collection[collection_name] = []

        # Rerank the combined results for debugging
        # reranked_combined_results = self.rerank_results(query, all_raw_results)

        # Deduplizieren der kombinierten rohen Ergebnisse basierend auf der ID
        deduplicated_raw_results = []
        seen_ids = set()
        for result in all_raw_results:
            if result.id not in seen_ids:
                deduplicated_raw_results.append(result)
                seen_ids.add(result.id)

        # Rerank die deduplizierten Ergebnisse
        reranked_deduplicated_results = self.rerank_results(
            query, deduplicated_raw_results
        )

        return {
            "raw_results_per_collection": raw_results_per_collection,
            "reranked_results_per_collection": reranked_results_per_collection,
            "deduplicated_combined_results": reranked_deduplicated_results,
        }


if __name__ == "__main__":
    collection_name_prefix = "test_collection"
    hybrid_search = Hybrid_search(collection_name_prefix)
    search_query = "Was soll im Datenschutz gemacht werden?"
    limit = 20

    hybrid_results = hybrid_search.query_hybrid_search(search_query, limit=limit)

    # print("\nRaw Results per Collection:")
    # for collection, results in hybrid_results["raw_results_per_collection"].items():
    #    print(f"  Collection: {collection}")
    #    for result in results:
    #        print(
    #            f"    ID: {result.id}, Score: {result.score}, Payload: {result.payload}"
    #        )

    # print("\nReranked Results per Collection:")
    # for collection, results in hybrid_results[
    #    "reranked_results_per_collection"
    # ].items():
    #    print(f"  Collection: {collection}")
    #    for result in results:
    #        print(
    #            f"    ID: {result.id}, Score: {result.score}, Payload: {result.payload}"
    #        )

    print("\nDeduplicated and Reranked Combined Results:")
    for result in hybrid_results["deduplicated_combined_results"]:
        print(f"  ID: {result.id}, Score: {result.score}, Payload: {result.payload}")
