
from tortoise import fields, Model

class ProductModel(Model):
    id = fields.TextField(pk=True)
    code = fields.TextField()
    name = fields.TextField()
    seller_name = fields.TextField()
    category_id = fields.TextField(null=True, blank=True)
    manufacturer_name = fields.TextField(null=False)
    short_description = fields.TextField(null=True, blank=True)
    long_description = fields.TextField(null=True, blank=True)
    country_of_origin = fields.TextField(null=True, blank=True)
    gross_weight = fields.FloatField(default=0.0)
    dimension = fields.TextField(null=True, blank=True)
    domain_category_code = fields.TextField(null=True, blank=True, default='')

    class Meta:
        table = "product"

    def get_text_for_embedding(self) -> str:
        """
        Get text fields for generating embeddings.
        Replaces None or missing values with an empty string.
        """
        fields = [
            ("Name", self.name),
            ("Seller", self.seller_name),
            ("Category", self.category_id),
            ("Manufacturer", self.manufacturer_name),
            ("Short Description", self.short_description),
            ("Long Description", self.long_description),
            ("Country of Origin", self.country_of_origin),
            ("Gross Weight", self.gross_weight),
            ("Dimension", self.dimension),
            ("Domain Category Code", self.domain_category_code)
        ]
        
        return " ".join(
            f"{label}: {value}" 
            for label, value in fields 
            if value is not None and value != ""
        )
    


class ProductAttributeModel(Model):
    id = fields.IntField(pk=True)
    product_code = fields.TextField()
    seller_id = fields.TextField()
    sku_id = fields.TextField()
    attribute_key = fields.TextField(null=True)
    attribute_value = fields.TextField(null=True)
    is_ondc_specific = fields.BooleanField(default=False)
    class Meta:
        table = "product_attribute"
    
    def get_text_for_embedding(self) -> str:
        """
        Get text fields for generating embeddings.
        """
    
        return f"{self.attribute_key}: {self.attribute_value}"
            