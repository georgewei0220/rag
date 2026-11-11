from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
import config


def get_client() -> QdrantClient:
    return QdrantClient(path=config.QDRANT_PATH)
    # 伺服器模式改成：QdrantClient(url="http://localhost:6333")


def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(config.COLLECTION):
        client.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=VectorParams(size=config.VECTOR_SIZE, distance=Distance.COSINE),
        )
