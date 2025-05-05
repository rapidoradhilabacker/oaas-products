# **Product Recommendation System**

This project is a **FastAPI-based product recommendation system** that uses **Elasticsearch** for vector search and **Sentence Transformers** for generating embeddings. It provides APIs for managing product data (bulk insert, add, update, delete) and fetching product recommendations based on product descriptions.

---

## **Features**
- **Bulk Insert Products**: Add multiple products at once.
- **Update a Product**: Update the description of an existing product.
- **Delete a Product**: Remove a product from the system.
- **Fetch Recommendations By Product ID**: Get product recommendations based on a product ID.
- **Fetch Recommendations By Query**: Get product recommendations based on a query string.

---

## **Technologies Used**
- **FastAPI**: For building the API.
- **Elasticsearch**: For storing and searching product embeddings.
- **Sentence Transformers**: For generating embeddings from product descriptions.
- **Pydantic**: For request/response validation.
- **Uvicorn**: For running the FastAPI server.

---

## **Project Structure**
```
product_recommendation/
│
├── main.py                  # FastAPI application entry point
├── config.py                # Configuration settings
├── constants.py             # Constants
├── database.py              # Database connection
├── utils.py                 # Utils
├── products/                # Products-related code
│   ├── embeddings.py        # Embedding generation using Sentence Transformers
│   ├── constants.py         # Constants
│   ├── routers.py           # API routers
│   ├── models.py            # Data models
│   ├── schemas.py           # Request and response models
│   ├── utils.py             # Utility functions
│   ├── config.py            # Configuration (e.g., Elasticsearch client)
│
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3.8 or higher.
- Elasticsearch installed and running on `http://localhost:9200`.

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

### **3. Run the Application**
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

---

## **API Documentation**

### Swagger UI


This will be available at `http://127.0.0.1:8000/docs`.

---

### **Base URL**
```
http://127.0.0.1:8000
```

---

### **Environment Variables**


- DB_PORT : Port of the database
- DB_IP : IP of the database
- DB_PASSWORD : Password of the database
- DB_DATABASE : Database name
- DB_USER : User of the database
- ELASTICSEARCH_URL : URL of the Elasticsearch
- ELASTICSEARCH_API_KEY : API key of the Elasticsearch
- ELASTICSEARCH_USERNAME : Username of the Elasticsearch
- ELASTICSEARCH_PASSWORD : Password of the Elasticsearch
- ELASTICSEARCH_INDEX : Index of the Elasticsearch


---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request. 
