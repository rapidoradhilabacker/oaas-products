from app.tracing import tracer
from app.product.models import ProductAttributeModel, ProductModel
from httpx import AsyncClient
from app.product.schemas import (
    InboundDocumentType,
    FolderDocumentInfo,
    FolderResponse
)
from app.product.openai_service import OpenAIService
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from fastapi import HTTPException
import os
import zipfile
from pdf2image import convert_from_bytes
import asyncio

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
    If ZIP archive, extract all image files. If PDF, convert each page to JPEG.
    Otherwise, treat as a single image file.
    Returns a tuple of (list of image bytes, list of filenames).
    """
    ctype = content_type.lower()
    # Handle ZIP archives
    if ctype == InboundDocumentType.ZIP:
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

    # Handle PDF documents
    if ctype == InboundDocumentType.PDF:
        try:
            # Convert each PDF page to a PIL Image
            pages = await asyncio.get_event_loop().run_in_executor(
                None, convert_from_bytes, file_bytes
            )
            if not pages:
                raise HTTPException(
                    status_code=400,
                    detail="PDF contains no pages or conversion failed"
                )
            images = []
            names = []
            for idx, page in enumerate(pages, start=1):
                buf = BytesIO()
                # Save page as JPEG
                page.save(buf, format="JPEG")
                images.append(buf.getvalue())
                names.append(f"page_{idx}.jpg")
            return images, names
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert PDF to images: {e}"
            )

    # Fallback: treat as single image file
    return [file_bytes], ["upload"]

async def process_product_zip(
    zip_content: bytes,
    open_ai_service: OpenAIService
) -> list[FolderResponse]:
    """
    Process a zip file containing product folders and extract product information
    """
    with tracer.start_as_current_span("process_product_zip") as span:
        zip_buffer = BytesIO(zip_content)
        folder_responses = []
        
        try:
            with zipfile.ZipFile(zip_buffer) as zip_ref:
                # Group files by their parent folders
                folder_files: dict[str, list[str]] = {}
                for file_path in zip_ref.namelist():
                    if file_path.endswith('/'):
                        continue
                    parent_folder = os.path.dirname(file_path).split('/')[-1]
                    if not parent_folder:
                        continue
                    if parent_folder not in folder_files:
                        folder_files[parent_folder] = []
                    folder_files[parent_folder].append(file_path)

                # Process each folder
                for folder_name, files in folder_files.items():
                    images = []
                    file_names = []
                    
                    # Extract images from the folder
                    for file_path in files:
                        if any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                            image_data = zip_ref.read(file_path)
                            images.append(image_data)
                            file_names.append(os.path.basename(file_path))
                    
                    if not images:
                        continue

                    # Get product info using OpenAI
                    try:
                        product_info = await open_ai_service.extract_product_info(images)
                        folder_info = FolderDocumentInfo(
                            product_code=folder_name,
                            product_name=product_info['product_name'],
                            short_description=product_info['short_description'],
                            long_description=product_info['long_description'],
                            file_type="image/jpeg",
                            file_name=file_names
                        )
                        
                        folder_response = FolderResponse(
                            folder=folder_name,
                            products=[folder_info]
                        )
                        folder_responses.append(folder_response)
                    except Exception as e:
                        print(f"Error processing folder {folder_name}: {str(e)}")
                        continue

        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=400,
                detail="Invalid zip file format"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing zip file: {str(e)}"
            )
            
        return folder_responses