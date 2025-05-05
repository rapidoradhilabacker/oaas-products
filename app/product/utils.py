from tortoise.exceptions import DoesNotExist
from app.tracing import tracer
from app.product.models import ProductAttributeModel, ProductModel

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