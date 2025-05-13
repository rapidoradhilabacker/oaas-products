import math
import numpy as np
from app.product.models import ProductModel
from app.product.schemas import ProductAttrData
from app.product.utils import get_text_for_embedding
from app.product.config import es, model, asy_es
from typing import Any
from elasticsearch import NotFoundError
from elasticsearch.helpers import async_bulk
from app.tracing import tracer


async def generate_embeddings(texts: str) -> np.ndarray:
    """
    Generate embeddings for a given text using a Sentence Transformer model.

    Args:
        texts (str): The input text for which embeddings are to be generated.

    Returns:
        np.ndarray: A numpy array containing the embeddings.
    """
    with tracer.start_as_current_span("generate_embeddings") as span:
        # Encode the text and return the embeddings as a numpy array
        return model.encode(texts, show_progress_bar=False)


async def upsert_embeddings_to_elasticsearch(prod_data: ProductAttrData, delete_all: bool = False):
    """
    Upsert product embeddings and metadata into Elasticsearch using the bulk API.
    """
    with tracer.start_as_current_span("upsert_embeddings_to_elasticsearch") as span:
        products: list[ProductModel] = prod_data.products
        if delete_all:
            await delete_all_embeddings_from_elasticsearch()

        actions = []
        for product in products:
            text = await get_text_for_embedding(product, prod_data.attribute_mapping.get(product.code, []))
            embedding = await generate_embeddings(text)
            gross_weight = product.gross_weight
            if gross_weight is not None and math.isnan(gross_weight):
                gross_weight = 0.0

            doc = {
                "id": product.id,
                "code": product.code,
                "name": product.name,
                "embedding": embedding.tolist()
            }
        
            action = {
                "_op_type": "index",
                "_index": ELASTICSEARCH_INDEX,
                "_id": product.id,
                "_source": doc
            }
            actions.append(action)

        try:
            await async_bulk(asy_es, actions)
            print("Bulk upsert completed successfully.")
        except Exception as e:
            print(f"Error during bulk upsert: {e}")

async def update_embedding_in_elasticsearch(prod_data: ProductAttrData):
    """
    Update a product's metadata in Elasticsearch.
    """
    with tracer.start_as_current_span("update_embedding_in_elasticsearch") as span:
        products: list[ProductModel] = prod_data.products
        for product in products:
            text = await get_text_for_embedding(product, prod_data.attribute_mapping.get(product.code, []))
            embedding = await generate_embeddings(text)
            doc = {
                "id": product.id,
                "code": product.code,
                "name": product.name,
                "embedding": embedding.tolist()
            }
            es.update(index=ELASTICSEARCH_INDEX, id=product.id, body={"doc": doc})

async def delete_embedding_from_elasticsearch(code: str):
    """
    Delete a product and its embeddings from Elasticsearch by ID.
    """
    with tracer.start_as_current_span("delete_embedding_from_elasticsearch") as span:
        es.delete(index=ELASTICSEARCH_INDEX, id=code)
        print(f"Deleted product with ID '{code}'.")

async def delete_embeddings_from_elasticsearch(codes: list[str]):
    """
    Delete multiple products and their embeddings from Elasticsearch by ID.
    """
    with tracer.start_as_current_span("delete_embeddings_from_elasticsearch") as span:
        response = es.delete_by_query(
            index=ELASTICSEARCH_INDEX,
            body={"query": {"terms": {"_code": codes}}}
        )
        print(f"Deleted {response['deleted']} documents.")

async def delete_all_embeddings_from_elasticsearch():
    """
    Delete all products and their embeddings from Elasticsearch.
    """
    with tracer.start_as_current_span("delete_all_embeddings_from_elasticsearch") as span:
        response = es.delete_by_query(
            index=ELASTICSEARCH_INDEX,
            body={"query": {"match_all": {}}}
        )
        print(f"Deleted {response['deleted']} documents.")

async def fetch_recommendations_from_elasticsearch(product_id: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Fetch recommendations for a given product ID using Elasticsearch's k-NN search.
    """
    with tracer.start_as_current_span("fetch_recommendations_from_elasticsearch") as span:
        # Fetch the embedding for the given product ID
        response = es.get(index=ELASTICSEARCH_INDEX, id=product_id)
        embedding = response['_source']['embedding']

        # Perform k-NN search using cosine similarity
        query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": embedding}
                    }
                }
            },
            "size": top_k
        }
        results = es.search(index=ELASTICSEARCH_INDEX, body=query)
        return [
            {
                "id": hit["_id"],
                "name": hit["_source"]["name"],
                "score": hit["_score"]
            }
            for hit in results["hits"]["hits"]
        ]

async def fetch_recommendations_from_elasticsearch_based_on_query(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Fetch recommendations for a given query using Elasticsearch's k-NN search.

    Args:
        query: The search query string.
        top_k: The number of recommendations to fetch.

    Returns:
        A list of recommendations, each containing the document ID, name, and score.
    """
    with tracer.start_as_current_span("fetch_recommendations_from_elasticsearch_based_on_query") as span:
        # Generate embeddings for the query
        embedding = await generate_embeddings(query)
        
        # Define the Elasticsearch query
        es_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": embedding}
                    }
                }
            },
            "size": top_k
        }
        
        # Execute the search
        results = es.search(index=ELASTICSEARCH_INDEX, body=es_query)
        
        # Format and return the results
        return [
            {
                "id": hit["_id"],
                "name": hit["_source"]["name"],
                "score": hit["_score"]
            }
            for hit in results["hits"]["hits"]
        ]