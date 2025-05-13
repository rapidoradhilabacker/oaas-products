import pytest
from unittest.mock import patch, MagicMock
from app.product.embeddings import (
    delete_all_embeddings_from_elasticsearch,
    delete_embedding_from_elasticsearch,
    fetch_recommendations_from_elasticsearch_based_on_query,
    update_embedding_in_elasticsearch,
    upsert_embeddings_to_elasticsearch,
    delete_embeddings_from_elasticsearch,
    fetch_recommendations_from_elasticsearch,
)

@pytest.fixture
def mock_es_client():
    with patch("app.product.embeddings.AsyncElasticsearch") as mock:
        mock_instance = MagicMock()
        # Setup common response patterns
        mock_instance.index.return_value = {"result": "created"}
        mock_instance.update.return_value = {"result": "updated"}
        mock_instance.delete.return_value = {"result": "deleted"}
        mock_instance.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.9,
                        "_source": {"name": "Test Product"}
                    }
                ]
            }
        }
        mock_instance.bulk.return_value = {"errors": False}
        yield mock_instance

@pytest.mark.asyncio
async def test_upsert_embeddings(mock_es_client):
    products = [
        {"id": "1", "name": "Test Product", "description": "Test Description"}
    ]
    attribute_mapping = {"1": {"attr1": "value1"}}
    
    await upsert_embeddings_to_elasticsearch(
        {"products": products, "attribute_mapping": attribute_mapping},
        delete_all=False
    )
    
    mock_es_client.bulk.assert_called_once()
    # Verify bulk operation arguments
    args = mock_es_client.bulk.call_args[0][0]
    assert len(args) > 0  # Should contain index operations

@pytest.mark.asyncio
async def test_update_embedding(mock_es_client):
    products = [
        {"id": "1", "name": "Test Product", "description": "Updated Description"}
    ]
    attribute_mapping = {"1": {"attr1": "updated_value"}}
    
    await update_embedding_in_elasticsearch(
        {"products": products, "attribute_mapping": attribute_mapping}
    )
    
    mock_es_client.update.assert_called_once()
    # Verify update operation arguments
    args = mock_es_client.update.call_args[1]
    assert args["id"] == "1"

@pytest.mark.asyncio
async def test_delete_embedding(mock_es_client):
    product_id = "1"
    
    await delete_embedding_from_elasticsearch(product_id)
    
    mock_es_client.delete.assert_called_once_with(
        index=pytest.approx(str),  # Any string for index name
        id=product_id
    )

@pytest.mark.asyncio
async def test_delete_embeddings(mock_es_client):
    product_ids = ["1", "2"]
    
    await delete_embeddings_from_elasticsearch(product_ids)
    
    # Should perform bulk delete operation
    mock_es_client.bulk.assert_called_once()
    args = mock_es_client.bulk.call_args[0][0]
    assert len(args) == len(product_ids) * 2  # Each deletion requires 2 operations

@pytest.mark.asyncio
async def test_delete_all_embeddings(mock_es_client):
    await delete_all_embeddings_from_elasticsearch()
    
    # Should perform delete_by_query operation
    mock_es_client.delete_by_query.assert_called_once_with(
        index=pytest.approx(str),
        body={"query": {"match_all": {}}}
    )

@pytest.mark.asyncio
async def test_fetch_recommendations(mock_es_client):
    product_id = "1"
    top_k = 5
    
    results = await fetch_recommendations_from_elasticsearch(product_id, top_k)
    
    mock_es_client.search.assert_called_once()
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["score"] == 0.9

@pytest.mark.asyncio
async def test_fetch_recommendations_by_query(mock_es_client):
    query = "test product"
    
    results = await fetch_recommendations_from_elasticsearch_based_on_query(query)
    
    mock_es_client.search.assert_called_once()
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["score"] == 0.9

@pytest.mark.asyncio
async def test_upsert_embeddings_with_delete_all(mock_es_client):
    products = [
        {"id": "1", "name": "Test Product", "description": "Test Description"}
    ]
    attribute_mapping = {"1": {"attr1": "value1"}}
    
    await upsert_embeddings_to_elasticsearch(
        {"products": products, "attribute_mapping": attribute_mapping},
        delete_all=True
    )
    
    # Should first delete all documents
    mock_es_client.delete_by_query.assert_called_once()
    # Then perform bulk insert
    mock_es_client.bulk.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling():
    with patch("app.product.embeddings.AsyncElasticsearch") as mock:
        mock_instance = MagicMock()
        mock_instance.index.side_effect = Exception("Test error")
        mock.return_value = mock_instance
        
        with pytest.raises(Exception) as exc_info:
            await update_embedding_in_elasticsearch({
                "products": [{"id": "1"}],
                "attribute_mapping": {"1": {}}
            })
        
        assert str(exc_info.value) == "Test error" 