from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.product.routers import router as products_router
from app.database import closeConnection, connectToDatabase, initialize_db_logger
from app.constants import API_DOC_DESCRIPTION

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    initialize_db_logger()
    await connectToDatabase()
    
    yield

    # Shutdown logic
    await closeConnection()

app = FastAPI(lifespan=lifespan)

# Add custom CORS middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# App metadata
app.title = "Product Recommendation API"
app.description = API_DOC_DESCRIPTION
app.version = "1.0"
app.docs_url = "/"
app.summary = "Fast API project for product recommendation"

# Include routers
app.include_router(products_router, prefix="/products", tags=["products"])