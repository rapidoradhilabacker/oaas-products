from elasticsearch import AsyncElasticsearch, Elasticsearch
from sentence_transformers import SentenceTransformer



# asy_es = AsyncElasticsearch(
#     hosts=[ELASTICSEARCH_URL],
#     basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
#     verify_certs=True,
#     ca_certs=ELASTICSEARCH_CERT_PATH,
#     timeout=30,
#     retry_on_timeout=True,
#     max_retries=3,
# )

# es = Elasticsearch(
#     hosts=[ELASTICSEARCH_URL],
#     basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
#     verify_certs=True,
#     ca_certs=ELASTICSEARCH_CERT_PATH,
#     timeout=30,
#     retry_on_timeout=True,
#     max_retries=3,
# )


# es.info()
asy_es = None
es = None

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')