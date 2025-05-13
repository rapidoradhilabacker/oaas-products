from tortoise.exceptions import DoesNotExist
from app.tracing import tracer
from app.product.models import ProductAttributeModel, ProductModel
from httpx import AsyncClient
from app.product.schemas import InboundDocumentType
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from typing import Tuple, List
from fastapi import HTTPException

async def get_products(product_codes: list[str]) -> list[ProductModel]:
    """
    Get products from the database.
    """
    with tracer.start_as_current_span("get_products") as span:
        if not product_codes:
            return await ProductModel.all()

        return await ProductModel.filter(code__in=product_codes)

async def get_product_attribute_mapping(product_code: list[str]) -> dict:
    """
    Get product attribute mapping from the database.
    """
    with tracer.start_as_current_span("get_product_attribute_mapping") as span:
        attributes = await ProductAttributeModel.filter(product_code__in=product_code)
        attribute_mapping = dict[str, list[ProductAttributeModel]]()
        for attr in attributes:
            attribute_mapping.setdefault(attr.product_code, []).append(attr)
        return attribute_mapping

async def get_text_for_embedding(product: ProductModel, attrs: list[ProductAttributeModel]) -> str:
    """
    Get text fields for generating embeddings.
    """
    with tracer.start_as_current_span("get_text_for_embedding") as span:
        product_text = product.get_text_for_embedding()
        attr_text = " ".join([attr.get_text_for_embedding() for attr in attrs])
        return f"{product_text} {attr_text}"

async def fetch_file_bytes(
    url: str, client: AsyncClient
) -> tuple[bytes, str, str]:
    """
    Fetches file content from a URL and returns (content bytes, content_type, filename).
    """
    response = await client.get(url)
    response.raise_for_status()
    content = response.content
    content_type = response.headers.get("Content-Type", "application/octet-stream")
    filename = url.rsplit("/", 1)[-1]
    return content, content_type, filename

async def extract_images(
    file_bytes: bytes, content_type: str
) -> tuple[list[bytes], list[str]]:
    """
    If zip archive, extract all image files; otherwise, return single file.
    """
    if content_type.lower() == InboundDocumentType.ZIP:
        try:
            with ZipFile(BytesIO(file_bytes)) as zf:
                image_names = [f for f in zf.namelist() if f.lower().endswith((
                    ".png", ".jpg", ".jpeg", ".gif", ".bmp"
                ))]
                if not image_names:
                    raise HTTPException(
                        status_code=400,
                        detail="No image files found in ZIP archive"
                    )
                images = [zf.read(name) for name in image_names]
                return images, image_names
        except BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP archive")
    # Non-zip: return raw bytes
    return [file_bytes], []