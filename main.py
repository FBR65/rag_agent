from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from tqdm import tqdm
import numpy as np
import secrets
import uuid

QDRANT_HOST = "10.84.0.7"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

collection_name='test_collection'
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT,timeout=QDRANT_TIMEOUT)
embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")



def query_hybrid_search(self, query, metadata_filter=None, limit=5):
    """
    Perform a hybrid search using dense and sparse embeddings.

    Args:
        query (str): The query string.
        metadata_filter (models.Filter, optional): A Qdrant filter object based on metadata. Defaults to None.
        limit (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        List[models.ScoredPoint]: A list of scored points based on the query and metadata filter.
    """
    # Embed the query using the dense embedding model
    response = self.embed_client.send([query])
    response_json = json.loads(response.text)            
    dense_query = response_json['data'][0]['embedding']

    # Embed the query using the sparse embedding model
    sparse_query = list(self.sparse_embedding_model.embed([query]))[0]

    results = self.qdrant_client.query_points(
        collection_name=self.collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
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
        query=models.FusionQuery(fusion=models.Fusion.RRF), #Reciprocal Rerank Fusion
    )
    
    # Extract the text from the payload of each scored point
    documents = [point.payload['text'] for point in results.points]

    return documents

if __name__ == '__main__':
    search = Hybrid_search(collection_name='RAG-bc7bc742cbe64a6cb4677ea7aa6334fa')
    query = "Was geschah mit dem Wolf?"
    file_names = "5c_MrchenRotkppchen.pdf"
    metadata_filter = search.metadata_filter(file_names)
    results = search.query_hybrid_search(query, metadata_filter)
    logger.info(f"Found {len(results)} results for query: {query}")

    reranking_instance = Reranking()
    reranked_documents = reranking_instance.rerank_documents(query=query,documents=results)
    print(reranked_documents)