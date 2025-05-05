from enum import Enum
from urllib.parse import quote, quote_plus
from pydantic import Field
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    """
    Configuration settings for the database connection.

    Attributes:
        port (int): The port number for the database.
        ip (str): The IP address or hostname of the database.
        password (str): The password for the database user.
        database (str): The name of the database.
        user (str): The username for the database. Defaults to 'postgres'.
    """
    port: int = Field(default=5433)
    ip: str = Field(default='live2')
    password: str = Field(default='postgres')
    database: str = Field(default='live_b2_ondc')
    user: str = Field(default='postgres')
    
    class Config:
        env_prefix = 'DB_'
    
    def get_database_url(self) -> str:
        """
        Generates a PostgreSQL connection URL using the provided settings.

        Returns:
            str: A PostgreSQL connection URL.
        """
        encoded_password = quote_plus(self.password)
        return f'postgres://{self.user}:{encoded_password}@{self.ip}:{self.port}/{self.database}'

# Instantiate the database settings
DB_SETTINGS = DatabaseSettings() 

class LogLevel(int, Enum):
    """
    Enumeration for log levels.

    Attributes:
        DEBUG (int): Debug level (10).
        INFO (int): Info level (20).
        WARNING (int): Warning level (30).
        ERROR (int): Error level (40).
        CRITICAL (int): Critical level (50).
    """
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    
class LogSettings(BaseSettings):
    """
    Configuration settings for logging.

    Attributes:
        level (int): The log level. Defaults to LogLevel.INFO.
        file_name (str): The file name for the logs. Defaults to ondc_log.
    """
    level: int = Field(default=LogLevel.INFO.value)
    file_name :str = Field(default='ondc_log')

    class Config:
        env_prefix = "LOG_"
    
class OpenAiSettings(BaseSettings):
    """
    Configuration settings for Open AI.

    Attributes:
        api_key (str): The api key for Open AI.
    """
    api_key: str = Field(default='')

    class Config:
        env_prefix = "OPENAI_"

# Instantiate the log settings
LOG_SETTINGS = LogSettings()
OPEN_AI_SETTINGS = OpenAiSettings()