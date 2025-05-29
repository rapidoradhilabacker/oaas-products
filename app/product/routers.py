import time
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, HTTPException, Query, Depends
import httpx
import asyncio
from app.product.openai_service import OpenAIService
from app.product.utils import get_product_attribute_mapping, get_products, fetch_file_bytes, extract_images, process_product_zip
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
    ZipProductRequest,
    MultiFolderResponse,
    FolderResponse,
    FolderDocumentInfo,
    CombinedProductRequest,
    InboundDocumentType,
    Product,
    ProductBytes,
    ImageBytes,
    Image,
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
from app.s3 import S3Service
from app.auth import get_current_user
import base64
from app.schemas import Trace

router = APIRouter()

# Dependency to provide a single shared HTTPX AsyncClient
async def get_http_client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(timeout=30.0) as client:
        yield client

@router.post("/bulk_insert/", response_model=BulkInsertResponse)
async def bulk_insert_products(request: BulkProductCreate, trace: Trace = Depends(get_current_user)):
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
async def update_product(request: ProductUpdate, trace: Trace = Depends(get_current_user)):
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
async def delete_products(request: ProductDelete, trace: Trace = Depends(get_current_user)):
    """
    Delete multiple products from Elasticsearch.
    """
    with tracer.start_as_current_span("delete_products") as span:
        # Delete documents from Elasticsearch
        await delete_embeddings_from_elasticsearch(request.codes)
        return ProductDeleteResponse(message="Products deleted successfully")

@router.delete("/delete_all_products/", response_model=ProductDeleteResponse)
async def delete_all_products(trace: Trace = Depends(get_current_user)):
    """
    Delete multiple products from Elasticsearch.
    """
    with tracer.start_as_current_span("delete_all_products") as span:
        # Delete documents from Elasticsearch
        await delete_all_embeddings_from_elasticsearch()
        return ProductDeleteResponse(message="Products deleted successfully")


@router.get("/recommendations/{product_id}", response_model=RecommendationsResponse)
async def get_recommendations(product_id: str, top_k: int = Query(5, ge=1), trace: Trace = Depends(get_current_user)):
    """
    Fetch product recommendations based on a product ID.
    """
    with tracer.start_as_current_span("get_recommendations") as span:
        recommendations = await fetch_recommendations_from_elasticsearch(product_id, top_k)
        return RecommendationsResponse(recommendations=recommendations)

@router.post("/recommendations/query/", response_model=RecommendationsResponse)
async def get_recommendations_by_query(request: ProductQuery, trace: Trace = Depends(get_current_user)):
    """
    Fetch product recommendations based on a query.
    """
    with tracer.start_as_current_span("get_recommendations_by_query") as span:
        recommendations = await fetch_recommendations_from_elasticsearch_based_on_query(request.query)
        return RecommendationsResponse(recommendations=recommendations)

@router.post(
    "/fetch/product/info/",
    response_model=MultiFolderResponse,
)
async def fetch_product_info(
    request: DocumentRequest,
    trace: Trace = Depends(get_current_user),
    client: AsyncClient = Depends(get_http_client),
):
    start_time = time.perf_counter()
    s3_service = S3Service()    
    if not request.products:
        raise HTTPException(status_code=400, detail="No products provided")

    # Map tmp_code -> (bytes, type, filename)
    products_file_map: dict[str, tuple[bytes, str, str]] = {}
    # Attempt fetching for each product until one succeeds
    s3_urls: dict[str, list[str]] = {}
    for product in request.products:
        s3_response = await s3_service.upload_to_s3_file(request.user, product, request.tenant)
        s3_urls[product.product_code] = s3_response.get('s3_urls', {}).get(product.product_code, [])
        for image in product.images:
            try:
                content, ctype, fname = await fetch_file_bytes(image.url, client)
                products_file_map[product.product_code] = (content, ctype, fname)
                break
            except httpx.HTTPError:
                continue
        if product.product_code not in products_file_map:
            # No valid URL for this product
            continue

    if not products_file_map:
        raise HTTPException(
            status_code=400,
            detail="Unable to fetch any file from provided URLs for all products"
        )

    # Extract images per product
    image_tasks: dict[str, tuple[list[bytes], list[str], str]] = {}
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
    folder_responses: list[FolderResponse] = []
    
    for tmp_code, (images, names, ctype) in image_tasks.items():
        try:
            raw = await open_ai_service.extract_product_info(images)
            folder_info = FolderDocumentInfo(
                product_code=tmp_code,
                product_name=raw['product_name'],
                short_description=raw['short_description'],
                long_description=raw['long_description'],
                file_type=ctype,
                file_name=names
            )
            folder_response = FolderResponse(
                folder=tmp_code,
                products=[folder_info]
            )
            folder_responses.append(folder_response)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed for {tmp_code}: {e}"
            )

    duration = round(time.perf_counter() - start_time, 2)
    
    if not folder_responses:
        return MultiFolderResponse(
            user=request.user,
            success=False,
            error="Failed to process any products",
            time_taken=duration,
            s3_response=s3_urls
        )

    return MultiFolderResponse(
        user=request.user,
        success=True,
        data=folder_responses,
        time_taken=duration,
        s3_response=s3_urls
    )

@router.post(
    "/fetch/products/info/zip/",
    response_model=MultiFolderResponse,
)
async def fetch_product_info_from_zip(
    request: ZipProductRequest,
    trace: Trace = Depends(get_current_user),
    client: AsyncClient = Depends(get_http_client),
):
    start_time = time.perf_counter()

    try:
        # Fetch the zip file first
        response = await client.get(request.products.url)
        response.raise_for_status()
        zip_content = response.content

        # Process zip file and upload to S3 concurrently
        async with S3Service() as s3_service:
            s3_task = asyncio.create_task(
                s3_service.upload_to_s3_zip(request.user, request.products, request.tenant)
            )
            
            openai_service = OpenAIService()
            process_task = asyncio.create_task(
                process_product_zip(zip_content, openai_service)
            )

            # Wait for both tasks to complete
            s3_response, folder_responses = await asyncio.gather(s3_task, process_task)
            s3_urls = s3_response.get('s3_urls', {})

        duration = round(time.perf_counter() - start_time, 2)
        
        if not folder_responses:
            return MultiFolderResponse(
                user=request.user,
                success=False,
                error="No valid product folders found in the zip file",
                time_taken=duration,
                s3_response=s3_urls
            )

        return MultiFolderResponse(
            user=request.user,
            success=True,
            data=folder_responses,
            time_taken=duration,
            s3_response=s3_urls
        )

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch zip file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.post(
    "/fetch/combined/products/info/",
    response_model=DocumentResponse,
)
async def fetch_info_from_combined_products(
    request: CombinedProductRequest,
    trace: Trace = Depends(get_current_user),
    client: AsyncClient = Depends(get_http_client),
):
    """
    Extract information from combined product images.
    """
    start_time = time.perf_counter()
    
    try:
        # Validate products count
        products_count = request.products.products_count
        if products_count <= 0:
            raise HTTPException(
                status_code=400,
                detail="Products count must be greater than 0"
            )
        
        # Fetch all images
        images_data = []
        file_names = []
        
        for image in request.products.images:
            try:
                content, content_type, filename = await fetch_file_bytes(image.url, client)
                if image.image_type.lower() == InboundDocumentType.ZIP:
                    ext_images, names = await extract_images(content, image.image_type)
                    images_data.extend(ext_images)
                    file_names.extend(names)
                else:
                    images_data.append(content)
                    file_names.append(filename)
            except Exception as e:
                print(f"Failed to fetch image {image.url}: {str(e)}")
                continue
        
        
        if not images_data:
            raise HTTPException(
                status_code=400,
                detail="Failed to fetch any images from provided URLs"
            )
        
        file_name_map = dict(zip(file_names, images_data))
        # Extract product information using OpenAI first
        openai_service = OpenAIService()
        product_info_list = await openai_service.extract_combined_product_info(
            images_data, products_count, file_names
        )
        

        s3_product_urls_map = {} 
        s3_products = []       
        async with S3Service() as s3_service:
            for product_info in product_info_list:
                code = product_info.get("product_code")
                files = product_info.get("file_names")
                
                if not code or not files:
                    continue
                
                images: list[ImageBytes] = []
                for file in files:
                    try:
                        image_content = file_name_map.get(file)
                        if not image_content:
                            continue
                        base64_bytes = base64.b64encode(image_content).decode('utf-8')
                        images.append(ImageBytes(image_name=file, image_type=InboundDocumentType.IMAGE, image_bytes=base64_bytes))
                    except Exception as e:
                        print(f"Failed to upload image to S3: {str(e)}")
                
                s3_products.append(ProductBytes(product_code=code, images=images))
            try:
                s3_response = {}
                if s3_products:
                    s3_response = await s3_service.upload_to_s3_file_bytes(request.user, s3_products, request.tenant)
                    s3_product_urls_map = s3_response.get('s3_urls', {})
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload to S3: {str(e)}"
                )
            
        document_info_list: list[DocumentInfo] = []
        for product_info in product_info_list:
            # Generate a unique product code if not provided
            if not product_info.get("product_code"):
                product_info["product_code"] = f"PROD-{len(document_info_list) + 1}"
                            
            # Create document info
            doc_info = DocumentInfo(
                product_code=product_info["product_code"],
                product_name=product_info["product_name"],
                short_description=product_info["short_description"],
                long_description=product_info["long_description"],
                file_type=InboundDocumentType.IMAGE,
                s3_urls=s3_product_urls_map.get(product_info["product_code"], [])
            )
            document_info_list.append(doc_info)
        
        duration = round(time.perf_counter() - start_time, 2)
        
        return DocumentResponse(
            user=request.user,
            success=True,
            data=document_info_list,
            time_taken=duration
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.post("/fetch/combined/products/info/from/invoice", response_model=DocumentResponse)
async def fetch_info_from_invoice(
    request: CombinedProductRequest,
    trace: Trace = Depends(get_current_user),
    client: AsyncClient = Depends(get_http_client),
):
    """
    Extract product information from invoice images or PDFs.
    This endpoint specializes in extracting product details including price information from invoices.
    """
    try:
        with tracer.start_as_current_span("fetch_info_from_invoice") as span:
            start_time = time.perf_counter()
                        
            # Fetch all images
            images_data = []
            file_names = []
            
            for image in request.products.images:
                try:
                    content, content_type, filename = await fetch_file_bytes(image.url, client)
                    images, names = await extract_images(content, image.image_type)
                    images_data.extend(images)
                    file_names.extend(names)
                except Exception as e:
                    print(f"Failed to fetch image {image.url}: {str(e)}")
                    continue
            
            
            if not images_data:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to fetch any images from provided URLs"
                )
            
            file_name_map = dict(zip(file_names, images_data))
            # Extract product information using OpenAI first
            openai_service = OpenAIService()
            product_info_list = await openai_service.extract_combined_product_info_from_invoice(
                request.user.company_name, images_data, file_names
            )
            

            s3_product_urls_map: dict[str, list[str]] = {} 
            document_info_list: list[DocumentInfo] = []
            for product_info in product_info_list:
                # Generate a unique product code if not provided
                if not product_info.get("product_code"):
                    product_info["product_code"] = f"PROD-{len(document_info_list) + 1}"
                                
                # Create document info
                doc_info = DocumentInfo(
                    product_code=product_info["product_code"],
                    product_name=product_info["product_name"],
                    short_description=product_info["short_description"],
                    long_description=product_info["long_description"],
                    file_type=InboundDocumentType.IMAGE,
                    s3_urls=s3_product_urls_map.get(product_info["product_code"], []),
                    price=float(product_info.get("price", 0.0))
                )
                document_info_list.append(doc_info)
            
            duration = round(time.perf_counter() - start_time, 2)
            
            return DocumentResponse(
                user=request.user,
                success=True,
                data=document_info_list,
                time_taken=duration
            )
            
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )