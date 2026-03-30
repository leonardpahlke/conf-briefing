"""ChromaDB collection management with Ollama embeddings."""

import contextlib

from conf_briefing.config import Config
from conf_briefing.query.chunker import Chunk

COLLECTION_NAME = "conf_briefing"
BATCH_SIZE = 50


def _get_embedding_function(config: Config):
    """Create ChromaDB Ollama embedding function."""
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    return OllamaEmbeddingFunction(
        model_name=config.query.embedding_model,
        url=f"{config.query.ollama_base_url}/api/embed",
    )


def _get_client(config: Config):
    """Create ChromaDB persistent client."""
    import chromadb

    return chromadb.PersistentClient(path=f"{config.data_dir}/chroma")


def build_index(config: Config, chunks: list[Chunk]) -> int:
    """Build or rebuild the vector index from chunks.

    Deletes the existing collection and creates a fresh one.
    Returns the number of chunks indexed.
    """
    client = _get_client(config)
    ef = _get_embedding_function(config)

    # Full rebuild: delete existing collection if present
    with contextlib.suppress(ValueError):
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch upsert
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        total += len(batch)
        print(f"[index]   Indexed {total}/{len(chunks)} chunks...")

    return total


def query_index(
    config: Config,
    query: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    track: str | None = None,
) -> list[dict]:
    """Query the vector index and return matching chunks with metadata.

    Returns a list of dicts with keys: id, text, metadata, distance.
    """
    client = _get_client(config)
    ef = _get_embedding_function(config)

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    k = top_k or config.query.top_k

    # Build where filter
    where = _build_where_filter(chunk_types, track)

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
    )

    # Flatten results into list of dicts
    hits = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        hits.append(
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            }
        )

    return hits


def _build_where_filter(chunk_types: list[str] | None, track: str | None) -> dict | None:
    """Build a ChromaDB where filter from optional constraints."""
    conditions = []

    if chunk_types:
        if len(chunk_types) == 1:
            conditions.append({"chunk_type": chunk_types[0]})
        else:
            conditions.append({"chunk_type": {"$in": chunk_types}})

    if track:
        conditions.append({"track": track})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
