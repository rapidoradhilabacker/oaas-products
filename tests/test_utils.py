import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.product.utils import (
    get_products,
    get_product_attribute_mapping,
    fetch_file_bytes,
    extract_images
)
import io
from PIL import Image
import numpy as np

@pytest.fixture
def mock_db_connection():
    with patch("app.product.utils.get_db_connection") as mock:
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                "id": "1",
                "name": "Test Product",
                "description": "Test Description",
                "category": "Test Category"
            }
        ]
        mock.return_value = mock_conn
        yield mock_conn

@pytest.mark.asyncio
async def test_get_products(mock_db_connection):
    product_codes = ["1", "2"]
    
    products = await get_products(product_codes)
    
    assert len(products) == 1
    assert products[0]["id"] == "1"
    assert products[0]["name"] == "Test Product"
    mock_db_connection.fetch.assert_called_once()

@pytest.mark.asyncio
async def test_get_product_attribute_mapping(mock_db_connection):
    product_codes = ["1"]
    
    mapping = await get_product_attribute_mapping(product_codes)
    
    assert "1" in mapping
    assert mapping["1"]["category"] == "Test Category"
    mock_db_connection.fetch.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_file_bytes():
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.content = b"test content"
    mock_response.raise_for_status = MagicMock()
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    content, ctype, fname = await fetch_file_bytes(
        "http://test.com/image.jpg",
        mock_client
    )
    
    assert content == b"test content"
    assert ctype == "image/jpeg"
    assert fname == "image.jpg"
    mock_client.get.assert_called_once_with("http://test.com/image.jpg")

@pytest.mark.asyncio
async def test_fetch_file_bytes_error():
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Connection error")
    
    with pytest.raises(Exception) as exc_info:
        await fetch_file_bytes("http://test.com/image.jpg", mock_client)
    
    assert str(exc_info.value) == "Connection error"

@pytest.mark.asyncio
async def test_extract_images_jpeg():
    # Create a test JPEG image
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    
    images, names = await extract_images(img_bytes, "image/jpeg")
    
    assert len(images) == 1
    assert isinstance(images[0], bytes)
    assert names == []  # Default behavior for single images

@pytest.mark.asyncio
async def test_extract_images_pdf():
    # Mock PDF content
    pdf_content = b"%PDF-1.4\n..."
    
    with patch("app.product.utils.convert_from_bytes") as mock_convert:
        # Mock the PDF to image conversion
        mock_page = MagicMock()
        mock_page.size = (100, 100)
        mock_convert.return_value = [mock_page]
        
        # Mock the page to bytes conversion
        img_buffer = io.BytesIO()
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
            img_buffer, format='JPEG'
        )
        mock_page.get_image.return_value = img_buffer.getvalue()
        
        images, names = await extract_images(pdf_content, "application/pdf")
        
        assert len(images) == 1
        assert isinstance(images[0], bytes)
        assert names == ["page_1.jpg"]
        mock_convert.assert_called_once_with(pdf_content)

@pytest.mark.asyncio
async def test_extract_images_unsupported_type():
    with pytest.raises(ValueError) as exc_info:
        await extract_images(b"test content", "text/plain")
    
    assert "Unsupported file type" in str(exc_info.value)

@pytest.mark.asyncio
async def test_extract_images_corrupt_file():
    with pytest.raises(Exception):
        await extract_images(b"corrupt image data", "image/jpeg") 