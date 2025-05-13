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
    tmp_code: str
    product_name: str
    short_description: str
    long_description: str
    file_type: InboundDocumentType
    file_name: list[str]

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
    tmp_code: str
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

class Product(BaseModel):
    tmp_code: str
    images: list[Image]

class DocumentRequest(BaseModel):
    user: User
    products: list[Product]
    tenant: str = 'placeorder'

class ZipImageInfo(BaseModel):
    image_type: str
    url: str

class ZipProductRequest(BaseModel):
    user: User
    products: ZipImageInfo
    tenant: str = 'placeorder'