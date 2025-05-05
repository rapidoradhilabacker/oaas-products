from io import BytesIO
import time
from typing import Optional
from fastapi import APIRouter, Form, HTTPException, Query, UploadFile
import httpx
from app.product.openai_service import OpenAIService
from app.product.utils import get_product_attribute_mapping, get_products
from app.tracing import tracer
from app.product.schemas import (
    DocumentInfo,
    DocumentResponse,
    ProductAttrData,
    BulkProductCreate,
    ProductDelete,
    ProductQuery,
    ProductUpdate,
    BulkInsertResponse,
    ProductUpdateResponse,
    ProductDeleteResponse,
    RecommendationsResponse,
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

router = APIRouter()

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

@router.post("/fetch/productinfo/", response_model=RecommendationsResponse)
async def fetch_productinfo(
        file: Optional[UploadFile] = None,
        file_url: Optional[str] = Form(None)
    ):
    start_time = time.time()
    """
    Fetch product info based on image.
    """
    with tracer.start_as_current_span("fetch_productinfo") as span:
        open_ai_service = OpenAIService()
        span.set_attribute("component", "document_extraction")
        span.add_event("Start processing request")

        image_bytes = None
        try:
            # Check if an uploaded file is provided
            if file:
                span.add_event("Uploaded file provided")
                image_bytes = await file.read()
                span.set_attribute("file.size", len(image_bytes))
            # Otherwise, check if a file URL is provided
            elif file_url:
                span.add_event("File URL provided")
                async with httpx.AsyncClient() as client:
                    response = await client.get(file_url)
                    span.set_attribute("http.status_code", response.status_code)
                    if response.status_code != 200:
                        span.add_event("Failed to retrieve file from URL")
                        raise HTTPException(status_code=400, detail="Unable to retrieve file from URL")
                    image_bytes = response.content
                    span.set_attribute("file.size", len(image_bytes))
            else:
                span.add_event("No file or URL provided")
                raise HTTPException(status_code=400, detail="No file or file URL provided")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            raise e

        # Ensure image_bytes is not empty
        if not image_bytes:
            span.add_event("Empty file content", {"error": True})
            raise HTTPException(status_code=400, detail="Empty file content")

        # Initialize the S3 file service for saving the image
        file_name = f"doc_{int(time.time())}.png"
        span.set_attribute("s3.file_name", file_name)

        # Wrap file bytes in a BytesIO if file object is not available
        if file is None:
            file_obj = BytesIO(image_bytes)
            file = UploadFile(filename=file_name, file=file_obj)
            span.add_event("Wrapped file bytes into BytesIO object")

        product_info = await open_ai_service.extract_product_info(image_bytes)
        span.add_event("Scheduled S3 upload and document extraction tasks")


        try:
            doc_info = DocumentInfo(**product_info)
            span.add_event("Converted extraction result to DocumentInfo model")
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error", True)
            raise HTTPException(status_code=500, detail="Invalid document information format")

        time_taken = time.time() - start_time
        span.set_attribute("time_taken", round(time_taken, 2))
        span.add_event("Completed processing", {"duration": round(time_taken, 2)})

        return DocumentResponse(
            success=True,
            data=doc_info,
            time_taken=round(time_taken, 2)
        )

        # recommendations = await fetch_recommendations_from_elasticsearch_based_on_query(request.query)
        # return RecommendationsResponse(recommendations=recommendations)
