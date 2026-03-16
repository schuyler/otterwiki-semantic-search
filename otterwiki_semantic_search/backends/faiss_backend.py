"""FAISS vector store backend with sidecar metadata."""

import fcntl
import json
import logging
import os
import threading

import numpy as np

from otterwiki_semantic_search.backends.base import VectorBackend
from otterwiki_semantic_search.embeddings.base import EmbeddingFunction

log = logging.getLogger(__name__)


class FAISSBackend(VectorBackend):
    """Vector store backend using FAISS with IndexFlatIP.

    FAISS stores only vectors and returns integer indices. Metadata is
    stored in a sidecar JSON file (embeddings.json) alongside the FAISS
    binary index file.

    The sidecar maps integer index positions to:
    - chunk ID
    - page_path
    - chunk text
    - metadata dict

    Uses IndexFlatIP (inner product similarity on normalized vectors).
    """

    def __init__(self, index_dir, embedding_fn, dimensionality=None):
        """Initialize the FAISS backend.

        Args:
            index_dir: Directory for storing the FAISS index and sidecar files.
            embedding_fn: An EmbeddingFunction instance for computing embeddings.
            dimensionality: Vector dimensionality. If None, uses embedding_fn.dimensionality.
        """
        import faiss

        self._index_dir = index_dir
        self._embedding_fn = embedding_fn
        self._dim = dimensionality or embedding_fn.dimensionality
        self._lock = threading.Lock()

        os.makedirs(index_dir, exist_ok=True)

        self._index_path = os.path.join(index_dir, "index.faiss")
        self._sidecar_path = os.path.join(index_dir, "embeddings.json")
        self._lock_path = os.path.join(index_dir, ".lock")

        # Load existing index or create new
        if os.path.exists(self._index_path) and os.path.exists(self._sidecar_path):
            loaded_index = faiss.read_index(self._index_path)
            loaded_sidecar = self._load_sidecar()
            if loaded_index.ntotal == len(loaded_sidecar):
                self._index = loaded_index
                self._sidecar = loaded_sidecar
            else:
                log.warning(
                    "FAISS index/sidecar mismatch in %s (index.ntotal=%d, sidecar len=%d); resetting.",
                    index_dir, loaded_index.ntotal, len(loaded_sidecar),
                )
                self._index = faiss.IndexFlatIP(self._dim)
                self._sidecar = []
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            self._sidecar = []  # list of entry dicts, indexed by FAISS position

    def _load_sidecar(self):
        """Load the sidecar metadata file with a shared (read) file lock."""
        try:
            lock_fd = open(self._lock_path, "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_SH)
                with open(self._sidecar_path, "r") as f:
                    return json.load(f)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self):
        """Persist the FAISS index and sidecar to disk with an exclusive file lock.

        Writes to .tmp files first then renames atomically (POSIX) to avoid
        leaving index and sidecar out of sync if a crash occurs mid-write.
        Sidecar is renamed before index so that a partial crash always leaves
        sidecar ahead of (or equal to) index, which the mismatch guard handles.
        """
        import faiss

        tmp_index = self._index_path + ".tmp"
        tmp_sidecar = self._sidecar_path + ".tmp"

        lock_fd = open(self._lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            faiss.write_index(self._index, tmp_index)
            with open(tmp_sidecar, "w") as f:
                json.dump(self._sidecar, f)
            os.rename(tmp_sidecar, self._sidecar_path)
            os.rename(tmp_index, self._index_path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def upsert(self, ids, texts, metadatas, embeddings=None):
        """Insert or update chunks in the FAISS index.

        If chunks with the same IDs already exist, they are removed first.
        Then new chunks are added at the end of the index.

        Args:
            ids: Chunk IDs.
            texts: Chunk texts.
            metadatas: Per-chunk metadata dicts.
            embeddings: Pre-computed embeddings. If None, computed via embedding_fn.
        """
        with self._lock:
            # Remove existing entries with these IDs
            existing_ids = {entry["id"] for entry in self._sidecar}
            ids_to_remove = set(ids) & existing_ids
            if ids_to_remove:
                self._remove_by_ids(ids_to_remove)

            # Compute embeddings if not provided
            if embeddings is None:
                embeddings = self._embedding_fn.embed(texts)

            vectors = np.array(embeddings, dtype=np.float32)

            # Add to FAISS index
            self._index.add(vectors)

            # Add sidecar entries
            for i, chunk_id in enumerate(ids):
                self._sidecar.append({
                    "id": chunk_id,
                    "text": texts[i],
                    "metadata": metadatas[i],
                })

            self._save()

    def _remove_by_ids(self, ids_to_remove):
        """Remove entries by chunk ID. Rebuilds the index.

        Must be called while holding self._lock.
        """
        import faiss

        if not ids_to_remove or not self._sidecar:
            return

        # Find positions to keep
        keep_positions = []
        new_sidecar = []
        for i, entry in enumerate(self._sidecar):
            if entry["id"] not in ids_to_remove:
                keep_positions.append(i)
                new_sidecar.append(entry)

        if len(keep_positions) == len(self._sidecar):
            return  # nothing to remove

        # Rebuild index with only kept vectors
        if keep_positions:
            old_vectors = np.array(
                [self._index.reconstruct(i) for i in keep_positions],
                dtype=np.float32,
            )
            new_index = faiss.IndexFlatIP(self._dim)
            new_index.add(old_vectors)
        else:
            new_index = faiss.IndexFlatIP(self._dim)

        self._index = new_index
        self._sidecar = new_sidecar

    def delete(self, where):
        """Delete chunks matching a metadata filter.

        Args:
            where: Dict of metadata key/value pairs. All must match.
        """
        with self._lock:
            ids_to_remove = set()
            for entry in self._sidecar:
                if all(entry["metadata"].get(k) == v for k, v in where.items()):
                    ids_to_remove.add(entry["id"])
            if ids_to_remove:
                self._remove_by_ids(ids_to_remove)
                self._save()

    def query(self, query_texts=None, query_embeddings=None, n_results=5):
        """Search for similar chunks.

        Args:
            query_texts: Text queries (will be embedded via embedding_fn).
            query_embeddings: Pre-computed query vectors.
            n_results: Number of results per query.

        Returns:
            Dict matching ChromaDB's result format:
            {ids, documents, metadatas, distances} — each a list of lists.
        """
        # Snapshot index and sidecar under the lock so a concurrent upsert/delete
        # cannot swap them between the ntotal check and the actual search.
        with self._lock:
            index = self._index
            sidecar = list(self._sidecar)  # shallow copy

        if index.ntotal == 0:
            empty = [[] for _ in range(len(query_texts or query_embeddings or [[]]))]
            return {"ids": empty, "documents": empty, "metadatas": empty, "distances": empty}

        if query_texts is not None:
            query_embeddings = self._embedding_fn.embed(query_texts)

        query_vectors = np.array(query_embeddings, dtype=np.float32)

        # Clamp n_results to available vectors
        k = min(n_results, index.ntotal)
        scores, indices = index.search(query_vectors, k)

        result_ids = []
        result_docs = []
        result_metas = []
        result_distances = []

        for q_idx in range(len(query_vectors)):
            q_ids = []
            q_docs = []
            q_metas = []
            q_dists = []
            for r_idx in range(k):
                faiss_idx = int(indices[q_idx][r_idx])
                if faiss_idx < 0:
                    continue  # FAISS returns -1 for missing results
                score = float(scores[q_idx][r_idx])
                entry = sidecar[faiss_idx]
                q_ids.append(entry["id"])
                q_docs.append(entry["text"])
                q_metas.append(entry["metadata"])
                # Convert inner product similarity to distance-like metric.
                # For normalized vectors, IP = cosine similarity.
                # distance = 1 - similarity, so lower = more similar.
                q_dists.append(1.0 - score)
            result_ids.append(q_ids)
            result_docs.append(q_docs)
            result_metas.append(q_metas)
            result_distances.append(q_dists)

        return {
            "ids": result_ids,
            "documents": result_docs,
            "metadatas": result_metas,
            "distances": result_distances,
        }

    def count(self):
        return self._index.ntotal

    def reset(self):
        """Delete all data and recreate empty index."""
        import faiss

        with self._lock:
            self._index = faiss.IndexFlatIP(self._dim)
            self._sidecar = []
            self._save()
