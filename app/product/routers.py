from io import BytesIO
import time
import zipfile
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, Depends
import httpx
import io
import os
from app.product.openai_service import OpenAIService
from app.product.utils import get_product_attribute_mapping, get_products
from app.tracing import tracer
from httpx import Timeout
from app.product.schemas import (
    BulkInsertResponse,
    DocumentRequest,
    DocumentInfo,
    DocumentResponse,
    ProductAttrData,
    BulkProductCreate,
    ProductDelete,
    ProductQuery,
    ProductUpdate,
    ProductUpdateResponse,
    ProductDeleteResponse,
    RecommendationsResponse,
    InboundDocumentType,
    ZipProductRequest,
    FolderDocumentInfo,
    FolderResponse,
    MultiFolderResponse,
)
from app.product.embeddings import (
    delete_all_embeddings_from_elasticsearch,
    delete_embedding_from_elasticsearch,
    fetch_recommendations_from_elasticsearch_based_on_query,
    update_embedding_in_elasticsearch,
    upsert_embeddings_to_elasticsearch,
    delete_embeddings_from_elasticsearch,
    fetch_recommendations_from_elasticsearch,
)
from httpx import AsyncClient

from app.product.utils import fetch_file_bytes, extract_images

router = APIRouter()

# Dependency to provide a single shared HTTPX AsyncClient
async def get_http_client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(timeout=30.0) as client:
        yield client

@router.post("/bulk_insert/", response_model=BulkInsertResponse)
async def bulk_insert_products(request: BulkProductCreate):
    """
    Bulk insert products into Elasticsearch.
    """
    with tracer.start_as_current_span("bulk_insert_products") as span:
        # Fetch products from the database
        start_time = time.time()
        products = await get_products(request.codes)
        attribute_mapping = await get_product_attribute_mapping(request.codes)
        delete_all = len(request.codes) == 0
        # Upsert embeddings to Elasticsearch
        await upsert_embeddings_to_elasticsearch(ProductAttrData(products=products, attribute_mapping=attribute_mapping), delete_all)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return BulkInsertResponse(message="Bulk insert successful")

@router.put("/update_product/", response_model=ProductUpdateResponse)
async def update_product(request: ProductUpdate):
    """
    Update a product's details in Elasticsearch.
    """
    with tracer.start_as_current_span("update_product") as span:
        # Fetch existing product
        products = await get_products(request.codes)
        attribute_mapping = await get_product_attribute_mapping(request.codes)
        await update_embedding_in_elasticsearch(ProductAttrData(products=products, attribute_mapping=attribute_mapping))
        return ProductUpdateResponse(message="Product updated successfully")

@router.delete("/delete_product/{product_id}", response_model=ProductDeleteResponse)
async def delete_product(product_id: str):
    """
    Delete a product from Elasticsearch.
    """
    with tracer.start_as_current_span("delete_product") as span:
        # Delete document from Elasticsearch
        await delete_embedding_from_elasticsearch(product_id)
        return ProductDeleteResponse(message="Product deleted successfully")

@router.delete("/delete_products/", response_model=ProductDeleteResponse)
async def delete_products(request: ProductDelete):
    """
    Delete multiple products from Elasticsearch.
    """
    with tracer.start_as_current_span("delete_products") as span:
        # Delete documents from Elasticsearch
        await delete_embeddings_from_elasticsearch(request.codes)
        return ProductDeleteResponse(message="Products deleted successfully")

@router.delete("/delete_all_products/", response_model=ProductDeleteResponse)
async def delete_all_products(request: ProductDelete):
    """
    Delete multiple products from Elasticsearch.
    """
    with tracer.start_as_current_span("delete_all_products") as span:
        # Delete documents from Elasticsearch
        await delete_all_embeddings_from_elasticsearch()
        return ProductDeleteResponse(message="Products deleted successfully")


@router.get("/recommendations/{product_id}", response_model=RecommendationsResponse)
async def get_recommendations(product_id: str, top_k: int = Query(5, ge=1)):
    """
    Fetch product recommendations based on a product ID.
    """
    with tracer.start_as_current_span("get_recommendations") as span:
        recommendations = await fetch_recommendations_from_elasticsearch(product_id, top_k)
        return RecommendationsResponse(recommendations=recommendations)

@router.post("/recommendations/query/", response_model=RecommendationsResponse)
async def get_recommendations_by_query(request: ProductQuery):
    """
    Fetch product recommendations based on a query.
    """
    with tracer.start_as_current_span("get_recommendations_by_query") as span:
        recommendations = await fetch_recommendations_from_elasticsearch_based_on_query(request.query)
        return RecommendationsResponse(recommendations=recommendations)

@router.post(
    "/fetch/product/info/",
    response_model=DocumentResponse,
)
async def fetch_product_info(
    request: DocumentRequest,
    client: AsyncClient = Depends(get_http_client),
):
    start_time = time.perf_counter()

    if not request.products:
        raise HTTPException(status_code=400, detail="No products provided")

    # Map tmp_code -> (bytes, type, filename)
    products_file_map: dict[str, tuple[bytes, str, str]] = {}

    # Attempt fetching for each product until one succeeds
    for product in request.products:
        for image in product.images:
            try:
                content, ctype, fname = await fetch_file_bytes(image.url, client)
                products_file_map[product.tmp_code] = (content, ctype, fname)
                break
            except httpx.HTTPError:
                continue
        if product.tmp_code not in products_file_map:
            # No valid URL for this product
            continue

    if not products_file_map:
        raise HTTPException(
            status_code=400,
            detail="Unable to fetch any file from provided URLs for all products"
        )

    # Extract images per product
    image_tasks: dict[str, tuple[list[bytes], list[str]]] = {}
    for tmp_code, (content, ctype, fname) in products_file_map.items():
        images, names = await extract_images(content, ctype)
        names = names or [fname]
        image_tasks[tmp_code] = (images, names, ctype)

    if not image_tasks:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract any images from provided files"
        )

    # Call AI service
    open_ai_service = OpenAIService()
    final_data: list[DocumentInfo] = []
    for tmp_code, (images, names, ctype) in image_tasks.items():
        try:
            raw = await open_ai_service.extract_product_info(images)
            info = DocumentInfo(
                **raw,
                tmp_code=tmp_code,
                file_name=names,
                file_type=ctype
            )
            final_data.append(info)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed for {tmp_code}: {e}"
            )

    duration = round(time.perf_counter() - start_time, 2)
    return DocumentResponse(
        user=request.user,
        success=True,
        data=final_data,
        time_taken=duration,
    )

@router.post(
    "/fetch/products/info/zip/",
    response_model=MultiFolderResponse,
)
async def fetch_product_info_from_zip(
    request: ZipProductRequest,
    client: AsyncClient = Depends(get_http_client),
):
    start_time = time.perf_counter()

    try:
        # Fetch the zip file
        response = await client.get(request.products.url)
        response.raise_for_status()
        zip_content = response.content

        # Create a BytesIO object from the zip content
        zip_buffer = BytesIO(zip_content)
        
        # Open the zip file
        folder_responses: list[FolderResponse] = []
        
        with zipfile.ZipFile(zip_buffer) as zip_ref:
            # Group files by their parent folders
            folder_files: dict[str, list[str]] = {}
            for file_path in zip_ref.namelist():
                if file_path.endswith('/'):  # Skip directories
                    continue
                parent_folder = os.path.dirname(file_path).split('/')[-1]
                if not parent_folder:  # Skip files in root
                    continue
                if parent_folder not in folder_files:
                    folder_files[parent_folder] = []
                folder_files[parent_folder].append(file_path)

            # Process each folder
            open_ai_service = OpenAIService()
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
                        tmp_code=folder_name,
                        product_name=product_info['product_name'],
                        short_description=product_info['short_description'],
                        long_description=product_info['long_description'],
                        file_type="image/jpeg",  # Default to JPEG, can be enhanced to detect actual type
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

        duration = round(time.perf_counter() - start_time, 2)
        
        if not folder_responses:
            return MultiFolderResponse(
                user=request.user,
                success=False,
                error="No valid product folders found in the zip file",
                time_taken=duration
            )

        return MultiFolderResponse(
            user=request.user,
            success=True,
            data=folder_responses,
            time_taken=duration
        )

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch zip file: {str(e)}"
        )
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
