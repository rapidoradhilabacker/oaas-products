import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from app.product.schemas import (
    BulkProductCreate,
    ProductUpdate,
    ProductQuery,
    DocumentRequest,
    ZipProductRequest,
)
from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_get_products():
    with patch("app.product.routers.get_products") as mock:
        mock.return_value = [
            {"id": "1", "name": "Test Product", "description": "Test Description"}
        ]
        yield mock

@pytest.fixture
def mock_get_product_attribute_mapping():
    with patch("app.product.routers.get_product_attribute_mapping") as mock:
        mock.return_value = {"1": {"attr1": "value1"}}
        yield mock

@pytest.fixture
def mock_elasticsearch_ops():
    with patch("app.product.routers.upsert_embeddings_to_elasticsearch") as mock_upsert, \
         patch("app.product.routers.update_embedding_in_elasticsearch") as mock_update, \
         patch("app.product.routers.delete_embedding_from_elasticsearch") as mock_delete, \
         patch("app.product.routers.delete_embeddings_from_elasticsearch") as mock_delete_many, \
         patch("app.product.routers.delete_all_embeddings_from_elasticsearch") as mock_delete_all, \
         patch("app.product.routers.fetch_recommendations_from_elasticsearch") as mock_fetch, \
         patch("app.product.routers.fetch_recommendations_from_elasticsearch_based_on_query") as mock_fetch_query:
        
        mock_upsert.return_value = None
        mock_update.return_value = None
        mock_delete.return_value = None
        mock_delete_many.return_value = None
        mock_delete_all.return_value = None
        mock_fetch.return_value = [{"id": "2", "score": 0.9}]
        mock_fetch_query.return_value = [{"id": "3", "score": 0.8}]
        
        yield {
            "upsert": mock_upsert,
            "update": mock_update,
            "delete": mock_delete,
            "delete_many": mock_delete_many,
            "delete_all": mock_delete_all,
            "fetch": mock_fetch,
            "fetch_query": mock_fetch_query
        }

@pytest.mark.asyncio
async def test_bulk_insert_products(mock_get_products, mock_get_product_attribute_mapping, mock_elasticsearch_ops):
    request = BulkProductCreate(codes=["1", "2"])
    response = client.post("/bulk_insert/", json=request.dict())
    
    assert response.status_code == 200
    assert response.json() == {"message": "Bulk insert successful"}
    mock_get_products.assert_called_once_with(["1", "2"])
    mock_elasticsearch_ops["upsert"].assert_called_once()

@pytest.mark.asyncio
async def test_update_product(mock_get_products, mock_get_product_attribute_mapping, mock_elasticsearch_ops):
    request = ProductUpdate(codes=["1"])
    response = client.put("/update_product/", json=request.dict())
    
    assert response.status_code == 200
    assert response.json() == {"message": "Product updated successfully"}
    mock_get_products.assert_called_once_with(["1"])
    mock_elasticsearch_ops["update"].assert_called_once()

@pytest.mark.asyncio
async def test_delete_product(mock_elasticsearch_ops):
    response = client.delete("/delete_product/1")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Product deleted successfully"}
    mock_elasticsearch_ops["delete"].assert_called_once_with("1")

@pytest.mark.asyncio
async def test_delete_products(mock_elasticsearch_ops):
    request = {"codes": ["1", "2"]}
    response = client.delete("/delete_products/", json=request)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Products deleted successfully"}
    mock_elasticsearch_ops["delete_many"].assert_called_once_with(["1", "2"])

@pytest.mark.asyncio
async def test_get_recommendations(mock_elasticsearch_ops):
    response = client.get("/recommendations/1?top_k=5")
    
    assert response.status_code == 200
    assert response.json() == {"recommendations": [{"id": "2", "score": 0.9}]}
    mock_elasticsearch_ops["fetch"].assert_called_once_with("1", 5)

@pytest.mark.asyncio
async def test_get_recommendations_by_query(mock_elasticsearch_ops):
    request = ProductQuery(query="test query")
    response = client.post("/recommendations/query/", json=request.dict())
    
    assert response.status_code == 200
    assert response.json() == {"recommendations": [{"id": "3", "score": 0.8}]}
    mock_elasticsearch_ops["fetch_query"].assert_called_once_with("test query")

@pytest.fixture
def mock_openai_service():
    with patch("app.product.routers.OpenAIService") as mock:
        mock_instance = MagicMock()
        mock_instance.extract_product_info.return_value = {
            "product_name": "Test Product",
            "short_description": "Short desc",
            "long_description": "Long desc"
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_http_client():
    with patch("httpx.AsyncClient") as mock:
        mock_response = MagicMock()
        mock_response.content = b"test content"
        mock_response.raise_for_status = MagicMock()
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_fetch_product_info(mock_openai_service, mock_http_client):
    request = DocumentRequest(
        user="test_user",
        products=[{"tmp_code": "123", "images": [{"url": "http://test.com/image.jpg"}]}]
    )
    
    with patch("app.product.routers.fetch_file_bytes") as mock_fetch:
        mock_fetch.return_value = (b"content", "image/jpeg", "test.jpg")
        with patch("app.product.routers.extract_images") as mock_extract:
            mock_extract.return_value = ([b"image_content"], ["test.jpg"], "image/jpeg")
            
            response = client.post("/fetch/product/info/", json=request.dict())
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["data"]) == 1
            assert data["data"][0]["product_name"] == "Test Product"

@pytest.mark.asyncio
async def test_fetch_product_info_from_zip(mock_openai_service, mock_http_client):
    request = ZipProductRequest(
        user="test_user",
        products={"url": "http://test.com/products.zip"}
    )
    
    # Create mock zip content
    import io
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.writestr('product1/image.jpg', b'test image content')
    
    mock_http_client.get.return_value.content = zip_buffer.getvalue()
    
    response = client.post("/fetch/products/info/zip/", json=request.dict())
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 1  # One folder processed
    assert data["data"][0]["folder"] == "product1" 