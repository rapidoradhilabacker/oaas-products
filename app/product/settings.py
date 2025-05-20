from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Elasticsearch Settings
    elasticsearch_url: str = Field(default="https://localhost:9200", env="ELASTICSEARCH_URL")
    elasticsearch_api_key: str | None = Field(default=None, env="ELASTICSEARCH_API_KEY")
    elasticsearch_username: str = Field(default="elastic", env="ELASTICSEARCH_USERNAME")
    elasticsearch_password: str = Field(default="changeme", env="ELASTICSEARCH_PASSWORD")
    elasticsearch_index: str = Field(default="product-recommendations", env="ELASTICSEARCH_INDEX")
    elasticsearch_cert_path: str = Field(default="http_ca.crt", env="ELASTICSEARCH_CERT_PATH")

    # S3 Settings
    s3_base_url: str = Field(default="https://devg4.rapidor.co", env="S3_BASE_URL")

    class Config:
        case_sensitive = False
        env_file = ".env"


settings = Settings()