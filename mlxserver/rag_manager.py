import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# RAGインデックスの保存場所
RAG_DIR = Path.home() / ".my_mlx_server" / "rag_db"
RAG_DIR.mkdir(exist_ok=True, parents=True)

INDEX_FILE = RAG_DIR / "index.npy"
CHUNKS_FILE = RAG_DIR / "chunks.json"

def get_rag_status() -> Dict[str, Any]:
    """RAGインデックスの状態を返す"""
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return {"indexed_chunks": 0, "indexed_files": []}
    
    try:
        vectors = np.load(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        indexed_files = sorted(list(set(chunk['source'] for chunk in chunks)))
        return {"indexed_chunks": len(chunks), "indexed_files": indexed_files}
    except Exception:
        return {"indexed_chunks": 0, "indexed_files": []}

def add_to_index(source_file: str, chunks: List[str], vectors: np.ndarray):
    """新しいチャンクとベクトルをインデックスに追加する"""
    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        existing_vectors = np.load(INDEX_FILE)
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
        
        all_vectors = np.concatenate([existing_vectors, vectors], axis=0)
        
        new_chunk_metadata = [{"text": text, "source": source_file} for text in chunks]
        all_chunks = existing_chunks + new_chunk_metadata
    else:
        all_vectors = vectors
        all_chunks = [{"text": text, "source": source_file} for text in chunks]

    np.save(INDEX_FILE, all_vectors)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

def search_index(query_vector: np.ndarray, top_k: int = 3) -> List[str]:
    """インデックスから類似度の高いチャンクを検索する"""
    status = get_rag_status()
    if status["indexed_chunks"] == 0:
        return []

    index_vectors = np.load(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks_metadata = json.load(f)

    # query_vectorの次元数とindex_vectorsの次元数を合わせる
    if query_vector.shape[1] != index_vectors.shape[1]:
        print(f"Warning: Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({index_vectors.shape[1]}). Skipping search.")
        return []

    similarities = cosine_similarity(query_vector, index_vectors)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    
    return [chunks_metadata[i]["text"] for i in top_k_indices]

def clear_index():
    """RAGインデックスをすべて削除する"""
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if CHUNKS_FILE.exists():
        CHUNKS_FILE.unlink()