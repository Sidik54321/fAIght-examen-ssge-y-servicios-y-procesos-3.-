from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List, Tuple

import chromadb

from ollama_api import embed


def _db_path() -> str:
    return os.path.join(os.getcwd(), "chroma_db")


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=_db_path())


def _collection(name: str):
    return _client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def collection_count(collection_name: str) -> int:
    col = _collection(collection_name)
    try:
        return int(col.count())
    except Exception:
        return 0


def ensure_seeded(
    collection_name: str, text: str, *, source: str = "seed_boxeo"
) -> Dict[str, Any]:
    if collection_count(collection_name) > 0:
        return {"seeded": False, "added": 0}
    res = ingest_text(collection_name, text, source=source)
    return {"seeded": True, **res}


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def ingest_text(
    collection_name: str, text: str, *, source: str = "manual"
) -> Dict[str, Any]:
    col = _collection(collection_name)
    chunks = chunk_text(text)
    if not chunks:
        return {"added": 0}
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embed(chunks)
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return {"added": len(chunks)}


def query(
    collection_name: str, question: str, *, k: int = 5
) -> List[Dict[str, Any]]:
    col = _collection(collection_name)
    q_emb = embed([question])[0]
    res = col.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"chunk": doc, "metadata": meta, "distance": dist})
    return out


def list_collections() -> List[str]:
    try:
        cols = _client().list_collections()
    except Exception:
        return []
    out: List[str] = []
    for c in cols:
        name = getattr(c, "name", None)
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return sorted(set(out))


def collection_preview(collection_name: str, *, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    col = _collection(collection_name)
    try:
        res = col.get(limit=int(limit), offset=int(offset), include=["documents", "metadatas"])
    except Exception:
        return []
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    out: List[Dict[str, Any]] = []
    for _id, doc, meta in zip(ids, docs, metas):
        out.append({"id": _id, "chunk": doc, "metadata": meta})
    return out


def delete_ids(collection_name: str, ids: List[str]) -> None:
    if not ids:
        return
    col = _collection(collection_name)
    col.delete(ids=ids)


def clear_collection(collection_name: str) -> int:
    col = _collection(collection_name)
    try:
        col.delete(where={})
        return 0
    except Exception:
        deleted = 0
        offset = 0
        while True:
            try:
                res = col.get(limit=1000, offset=offset, include=[])
            except Exception:
                break
            ids = res.get("ids") or []
            if not ids:
                break
            col.delete(ids=ids)
            deleted += len(ids)
            offset += len(ids)
        return deleted
