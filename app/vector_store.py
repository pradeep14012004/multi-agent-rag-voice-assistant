"""Persistent Chroma vector store with deterministic IDs and duplicate-safe retrieval."""
import hashlib
import re
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME, MIN_RELEVANCE_SCORE

_model = None
_client = None
_collection = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def _stable_id(item: dict) -> str:
    m = item["metadata"]
    raw = "|".join([
        m.get("filename", ""), m.get("domain", ""), str(m.get("page", 1)),
        str(m.get("chunk_index", 0)), item["text"],
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def add_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    documents = {
        (item["metadata"]["filename"], item["metadata"]["domain"])
        for item in chunks
    }
    unique = {}
    for item in chunks:
        item = dict(item)
        item["id"] = _stable_id(item)
        unique[item["id"]] = item
    items = list(unique.values())
    model, collection = get_model(), get_collection()
    texts = [x["text"] for x in items]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    for filename, domain in documents:
        collection.delete(where={
            "$and": [
                {"filename": filename},
                {"domain": domain},
            ],
        })

    collection.upsert(
        ids=[x["id"] for x in items], documents=texts,
        metadatas=[x["metadata"] for x in items], embeddings=vectors,
    )
    return len(items)

def search(query: str, top_k: int = 5, domain: str | None = None, min_score: float = MIN_RELEVANCE_SCORE) -> list[dict]:
    query = query.strip()
    if not query or top_k <= 0:
        return []
    collection = get_collection()
    candidate_count = collection.count()
    if candidate_count == 0:
        return []

    query_vector = get_model().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).tolist()[0]
    kwargs = {
        "query_embeddings": [query_vector],
        "n_results": candidate_count,
        "include": ["documents", "metadatas", "distances"],
    }
    if domain and domain != "All domains":
        kwargs["where"] = {"domain": domain}
    result = collection.query(**kwargs)

    hits = []
    seen = set()
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for doc, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        distance = float(distance)
        score = 1.0 - distance
        if score < min_score:
            continue

        fingerprint = re.sub(r"\s+", " ", doc).strip().casefold()
        duplicate_key = (metadata.get("domain", ""), fingerprint)
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        hits.append({"text": doc, "metadata": metadata,
                     "distance": distance, "score": score})
        if len(hits) >= top_k:
            break
    return hits

def collection_count() -> int:
    return get_collection().count()

def reset_collection():
    global _client, _collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _client, _collection = client, None
