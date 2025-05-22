from pydantic import BaseModel
from typing import Any, Optional, List
from enum import Enum
from app.product.models import ProductAttributeModel, ProductModel

class BulkProductCreate(BaseModel):
    codes: list[str] = []

class ProductUpdate(BaseModel):
    codes: list[str] = []

class ProductDelete(BaseModel):
    codes: list[str] = []

class ProductQuery(BaseModel):
    query: str

# Response Models
class ProductResponse(BaseModel):
    id: str
    description: str

class BulkInsertResponse(BaseModel):
    message: str

class ProductAddResponse(BaseModel):
    message: str

class ProductUpdateResponse(BaseModel):
    message: str

class ProductDeleteResponse(BaseModel):
    message: str

class RecommendationsResponse(BaseModel):
    recommendations: list[dict[str, Any]]

class ProductAttrData(BaseModel):
    products: list[Any]
    attribute_mapping: dict[str, list[Any]]

class InboundDocumentType(str, Enum):
    PDF = "application/pdf"
    IMAGE = "image/jpeg"
    ZIP = "application/zip"
    JSON = "application/json"
    PNG = "image/png"
    BINARY = "binary/octet-stream"

class User(BaseModel):
    mobile_no: str  
    company_name: Optional[str] = ""
    
class DocumentInfo(BaseModel):
    product_code: str
    product_name: str
    short_description: str
    long_description: str
    file_type: InboundDocumentType
    s3_urls: list[str]
    price: float = 0.0

class DocumentResponse(BaseModel):
    user: User
    success: bool
    data:  list[DocumentInfo] = []
    error: Optional[str] = None
    time_taken: float

# --- Schemas ---
class FolderRequest(BaseModel):
    path: str  # Absolute or relative path to the product folder

class FolderDocumentInfo(BaseModel):
    product_code: str
    product_name: str
    short_description: str
    long_description: str
    file_type: str  # MIME type or custom InboundDocumentType name
    file_name: List[str]

class FolderResponse(BaseModel):
    folder: str
    products: List[FolderDocumentInfo]

class MultiFolderResponse(BaseModel):
    user: User
    success: bool
    data: List[FolderResponse] = []
    s3_response: dict[str, Any] = {}
    error: Optional[str] = None
    time_taken: float

class DocumentListResponse(BaseModel):
    success: bool
    data: Optional[List[DocumentInfo]] = None
    error: Optional[str] = None
    time_taken: float

class Image(BaseModel):
    image_type: InboundDocumentType
    url: str

class CombinedProductsImages(BaseModel):
    products_count: int
    images: list[Image]

class CombinedProductRequest(BaseModel):
    user: User
    products: CombinedProductsImages
    tenant: str = 'placeorder'

class Product(BaseModel):
    product_code: str
    images: list[Image]

class DocumentRequest(BaseModel):
    user: User
    products: list[Product]
    tenant: str = 'placeorder'

class ZipImageInfo(BaseModel):
    url: str

class ZipProductRequest(BaseModel):
    user: User
    products: ZipImageInfo
    tenant: str = 'placeorder'

class S3UploadZipRequest(BaseModel):
    user: User
    zip_folder: ZipImageInfo
    tenant: str = 'placeorder'

class S3UploadFileRequest(BaseModel):
    user: User
    product: Product
    tenant: str = 'placeorder'

class S3UploadResponse(BaseModel):
    success: bool
    data: dict[str, Any]
    error: Optional[str] = None
    time_taken: float

class ImageBytes(BaseModel):
    image_name: str
    image_type: InboundDocumentType
    image_bytes: str

class ProductBytes(BaseModel):
    product_code: str
    images: list[ImageBytes]

class S3UploadFileBytesRequest(BaseModel):
    user: User
    products: list[ProductBytes]
    tenant: str = 'placeorder'
