import logging
import sys
from tortoise import Tortoise

from app.config import DB_SETTINGS


async def connectToDatabase():
    await Tortoise.init(
        db_url=DB_SETTINGS.get_database_url(),
        modules={"models": ["app.product.models"]},
    )

async def closeConnection():
    await Tortoise.close_connections()

def initialize_db_logger():
    from app.utils import get_file_formatter, get_stdout_formatter
    from app.config import LOG_SETTINGS
    log_level = LOG_SETTINGS.level
    file_fmt = get_file_formatter()
    std_fmt = get_stdout_formatter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(std_fmt)
    sh.setLevel(log_level)
    logger_db_client = logging.getLogger("tortoise.db_client")
    logger_db_client.setLevel(log_level)
    logger_db_client.addHandler(sh)
    fh = logging.FileHandler(filename=LOG_SETTINGS.file_name)
    fh.setFormatter(file_fmt)
    logger_tortoise = logging.getLogger("tortoise")
    logger_tortoise.setLevel(logging.INFO)
    logger_tortoise.addHandler(fh)


