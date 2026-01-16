import os
import numpy as np
import faiss
import pyarrow.dataset as ds


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_ivfpq_from_local_parquet(
    parquet_dir: str,
    index_name: str = "text",
    dim: int = 100,
    nlist: int = 4096,
    m: int = 20,
    nbits: int = 8,
    train_size: int = 300_000,
    batch_size: int = 100_000,
    seed: int = 0,  
):
    """
    Streams Parquet batches and builds a FAISS IVF-PQ index without loading all vectors into RAM.
    Expects Parquet columns: doc_id (int), embedding (list[float]) where len(embedding)=dim.
    Writes:
      - {out_root}/{index_name}_vector/{index_name}_index.faiss
      - {out_root}/{index_name}_vector/doc_ids.npy (memmap-backed during build)
    """

    out_index_path = f"data/embeddings/{index_name}_vector/{index_name}_index.faiss"
    out_docids_path = f"data/embeddings/{index_name}_vector/doc_ids.npy"
    os.makedirs(os.path.dirname(out_index_path), exist_ok=True)

    rng = np.random.default_rng(seed)

    dataset = ds.dataset(parquet_dir, format="parquet")

    # Pass 1: reservoir-sample training vectors
    train_buf = []
    seen = 0

    scanner = dataset.scanner(columns=["embedding"], batch_size=batch_size)
    for batch in scanner.to_batches():
        emb_list = batch.column("embedding").to_pylist()
        if not emb_list:
            continue

        emb = np.array(emb_list, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] != dim:
            raise ValueError(f"dim mismatch: expected {dim}, got {emb.shape[1]}")

        for v in emb:
            seen += 1
            if len(train_buf) < train_size:
                train_buf.append(v)
            else:
                j = rng.integers(0, seen)
                if j < train_size:
                    train_buf[j] = v

        if len(train_buf) >= train_size and seen >= train_size * 5:
            break

    if not train_buf:
        raise RuntimeError("No embeddings found in Parquet.")

    train = l2_normalize(np.vstack(train_buf).astype(np.float32))

    # Build + train IVF-PQ
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(train)

    # Pass 2: count rows (doc_id only) to size memmap
    total = 0
    for b in dataset.scanner(columns=["doc_id"], batch_size=batch_size).to_batches():
        total += len(b)

    if total == 0:
        raise RuntimeError("No doc_id rows found in Parquet.")

    doc_ids_mm = np.memmap(out_docids_path, dtype=np.int32, mode="w+", shape=(total,))
    offset = 0

    scanner = dataset.scanner(columns=["doc_id", "embedding"], batch_size=batch_size)
    for batch in scanner.to_batches():
        doc_ids = np.array(batch.column("doc_id").to_pylist(), dtype=np.int32)
        emb_list = batch.column("embedding").to_pylist()
        if not emb_list:
            continue

        emb = np.array(emb_list, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        if len(doc_ids) != emb.shape[0]:
            raise ValueError(f"row mismatch: doc_ids={len(doc_ids)} emb_rows={emb.shape[0]}")

        if emb.shape[1] != dim:
            raise ValueError(f"dim mismatch: expected {dim}, got {emb.shape[1]}")

        emb = l2_normalize(emb)

        index.add(emb)
        doc_ids_mm[offset:offset + len(doc_ids)] = doc_ids
        offset += len(doc_ids)

    doc_ids_mm.flush()

    if offset != total:
        # If you ever skip rows, you must fix mapping. Better to fail loudly.
        raise RuntimeError(f"Added {offset} vectors but expected {total}. Mapping would be wrong.")

    faiss.write_index(index, out_index_path)

    return {"ntotal": int(index.ntotal), "index_path": out_index_path, "docids_path": out_docids_path}


if __name__ == "__main__":
    text_embeddings_path = "data/embeddings/text/"
    title_embeddings_path = "data/embeddings/title/"

    print(build_ivfpq_from_local_parquet(parquet_dir=text_embeddings_path, index_name="text"))
    print(build_ivfpq_from_local_parquet(parquet_dir=title_embeddings_path, index_name="title"))
