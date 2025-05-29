from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Elasticsearch Settings
    elasticsearch_url: str = Field(default="https://localhost:9200", alias="ELASTICSEARCH_URL")
    elasticsearch_api_key: str = Field(default="", alias="ELASTICSEARCH_API_KEY")
    elasticsearch_username: str = Field(default="elastic", alias="ELASTICSEARCH_USERNAME")
    elasticsearch_password: str = Field(default="changeme", alias="ELASTICSEARCH_PASSWORD")
    elasticsearch_index: str = Field(default="product-recommendations", alias="ELASTICSEARCH_INDEX")
    elasticsearch_cert_path: str = Field(default="http_ca.crt", alias="ELASTICSEARCH_CERT_PATH")

    # S3 Settings
    s3_base_url: str = Field(default="https://devg4.rapidor.co", alias="S3_BASE_URL")
    s3_auth_token: str = Field(default="", alias="S3_AUTH_TOKEN")

    class Config:
        case_sensitive = False
        env_file = ".env"


settings = Settings()