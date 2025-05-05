from pydantic import BaseModel
from typing import Any, Optional

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

class DocumentInfo(BaseModel):
    product_name: str
    product_code: str
    short_description: str
    long_description: str

class DocumentResponse(BaseModel):
    success: bool
    data: Optional[DocumentInfo] = None
    error: Optional[str] = None
    time_taken: float