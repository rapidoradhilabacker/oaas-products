from fastapi import HTTPException
import httpx
from app.product.schemas import (
    User,
    ZipImageInfo,
    S3UploadRequest
)
from app.product.openai_service import OpenAIService
from app.tracing import tracer
from app.product.settings import settings


class S3Service:
    """Service for handling S3-related operations"""
    
    def __init__(self):
        self.s3_upload_url = f"{settings.s3_zip_folder_url}/s3/upload/oaas/folder"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.openai_service = OpenAIService()

    async def upload_to_s3(self, user: User, zip_info: ZipImageInfo, tenant: str) -> dict:
        """
        Upload zip file to S3 asynchronously
        
        Args:
            user: User information
            zip_info: Information about the zip file to upload
            tenant: Tenant identifier
            
        Returns:
            dict: Response from S3 upload service
            
        Raises:
            HTTPException: If upload fails
        """
        with tracer.start_as_current_span("upload_to_s3") as span:
            s3_request = S3UploadRequest(
                user=user,
                zip_folder=ZipImageInfo(
                    url=zip_info.url
                ),
                tenant=tenant
            )

            try:
                response = await self.client.post(
                    self.s3_upload_url,
                    json=s3_request.model_dump(),
                    headers={"accept": "application/json", "Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to upload to S3: {str(e)}"
                )
            finally:
                await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
