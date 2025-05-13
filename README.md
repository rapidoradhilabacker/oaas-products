# **Product Recommendation and Information Extraction System**

This project is a **FastAPI-based system** that combines product recommendations using **Elasticsearch** for vector search and **OpenAI** for product information extraction from images. It provides APIs for managing product data, fetching recommendations, and extracting product information from images and ZIP files.

---

## **Features**
- **Product Management**:
  - Bulk insert products into Elasticsearch
  - Update existing products
  - Delete single or multiple products
  - Delete all products
- **Product Recommendations**:
  - Get recommendations by product ID
  - Get recommendations by query string
- **Product Information Extraction**:
  - Extract product info from individual images
  - Process ZIP files containing multiple product folders
  - AI-powered extraction of product name, descriptions using OpenAI

---

## **Technologies Used**
- **FastAPI**: For building the API
- **Elasticsearch**: For storing and searching product embeddings
- **OpenAI**: For AI-powered product information extraction
- **HTTPX**: For async HTTP requests
- **Pydantic**: For request/response validation
- **Zipfile**: For ZIP file processing
- **OpenTelemetry**: For distributed tracing

---

## **Project Structure**
```
oaas-products/
│
├── app/                     # Main application code
│   ├── product/            # Product-related code
│   │   ├── routers.py      # API endpoints
│   │   ├── schemas.py      # Data models
│   │   ├── embeddings.py   # Elasticsearch operations
│   │   ├── openai_service.py # OpenAI integration
│   │   └── utils.py        # Utility functions
│   └── tracing.py          # Tracing configuration
│
├── tests/                  # Test cases
│   ├── test_routers.py    # API endpoint tests
│   ├── test_embeddings.py # Elasticsearch operation tests
│   └── test_utils.py      # Utility function tests
│
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.8 or higher
- Elasticsearch installed and running
- OpenAI API key
- Docker (optional, for containerized deployment)

### **2. Install Dependencies**
1. Clone the repository:
   ```bash
   git clone https://github.com/adhilabu/oaas-products.git
   cd oaas-products
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **3. Environment Variables**
Create a `.env` file with the following variables:
```env
# Database Configuration
DB_PORT=5432
DB_IP=localhost
DB_PASSWORD=your_password
DB_DATABASE=your_database
DB_USER=your_user

# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_API_KEY=your_api_key
ELASTICSEARCH_USERNAME=your_username
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_INDEX=your_index

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

### **4. Run the Application**
Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

---

## **API Documentation**

### **Base URL**
```
http://127.0.0.1:8000
```

### **Key Endpoints**

#### Product Management
- `POST /bulk_insert/`: Bulk insert products
- `PUT /update_product/`: Update a product
- `DELETE /delete_product/{product_id}`: Delete a single product
- `DELETE /delete_products/`: Delete multiple products
- `DELETE /delete_all_products/`: Delete all products

#### Recommendations
- `GET /recommendations/{product_id}`: Get recommendations by product ID
- `POST /recommendations/query/`: Get recommendations by query

#### Product Information Extraction
- `POST /fetch/product/info/`: Extract info from product images
- `POST /fetch/products/info/zip/`: Process ZIP file containing product folders

For detailed API documentation, visit the Swagger UI at `http://127.0.0.1:8000/docs`.

---

## **Testing**
Run the test suite:
```bash
pytest tests/
```

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

---

## **License**
This project is licensed under the MIT License - see the LICENSE file for details. 
