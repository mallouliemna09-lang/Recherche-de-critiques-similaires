import os
import numpy as np
import pandas as pd
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN = 512

def strip_html(x):
    if not isinstance(x, str):
        return ""
    return BeautifulSoup(x, "html.parser").get_text(separator=" ", strip=True)


def chunk_by_tokens(text, tokenizer, max_len=512, stride=400):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False
    )
    ids = enc["input_ids"]

    n = len(ids)
    start = 0
    while start < n:
        end = start + max_len
        chunk_ids = ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            yield chunk_text
        start += stride



def build_vector_database(csv_path="interstellar_critique.csv"):

    df = pd.read_csv(csv_path)

    # full_review = titre + contenu sans HTML
    full_review_raw = df["review_title"].fillna("") + " . " + df["review_content"].fillna("")
    df["full_review"] = full_review_raw.map(strip_html)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = SentenceTransformer(MODEL)
    model.max_seq_length = MAX_LEN

    all_chunks = []
    meta_rows = []

    for _, row in df.iterrows():
        text = row["full_review"]
        id = str(row["id"])
        
        toks = tokenizer.tokenize(text)

        if len(toks) < MAX_LEN:
            all_chunks.append(text)
            meta_rows.append({
                "chunk_id": f"{id}_0",
                "id": id,
                "username": row["username"],
                "full_review": row["full_review"]
            })
        else:
            for i, chunk in enumerate(chunk_by_tokens(text, tokenizer, max_len=MAX_LEN)):
                all_chunks.append(chunk)
                meta_rows.append({
                        "chunk_id": f"{id}_chunk{i}",
                        "id": id,
                        "username": row["username"],
                        "full_review": row["full_review"]
                })

    emb = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    np.save("embeddings.npy",emb)
    faiss.write_index(index, "faiss.index")
    pd.DataFrame(meta_rows).to_csv("meta_chunks.csv", index=False, encoding="utf-8")


def embed_query(query_text, model, tokenizer):
   
    toks = tokenizer.tokenize(query_text)
    if len(toks) < MAX_LEN:
        vec = model.encode([query_text], normalize_embeddings=True)
        return vec.astype("float32")

    parts = list(chunk_by_tokens(query_text, tokenizer, max_len=MAX_LEN))
    part_vecs = model.encode(parts, normalize_embeddings=True)
    mean_vec = np.mean(part_vecs, axis=0, keepdims=True)
    return mean_vec.astype("float32")


def load_resources():
    chunks_df = pd.read_csv("meta_chunks.csv")
    index = faiss.read_index("faiss.index")
    model = SentenceTransformer(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    return chunks_df, index, model, tokenizer



def search_similarity_3(query_text, k, chunks_df, index, model, tokenizer):
    toks = tokenizer.tokenize(query_text)

    if len(toks) <= MAX_LEN:
        q_vec = model.encode([query_text], normalize_embeddings=True).astype("float32")
        D, I = index.search(q_vec, k)

        candidates = [(float(score), int(idx)) for score, idx in zip(D[0], I[0])]
    else:
        # chercher chunk par chunk
        parts = list(chunk_by_tokens(query_text, tokenizer, max_len=MAX_LEN))
        part_vecs = model.encode(parts, normalize_embeddings=True).astype("float32")

        # on agrège tous les résultats des chunks
        candidates = []
        for vec in part_vecs:
            vec = vec.reshape(1, -1) 
            D, I = index.search(vec, k)
            for score, idx in zip(D[0], I[0]):
                candidates.append((float(score), int(idx)))

        # Tri par score desc
        candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen = set()
    for score, idx in candidates:
        chunk_meta = chunks_df.iloc[idx]
        orig_id = chunk_meta["id"]
        if orig_id in seen:
            continue

        results.append({
            "orig_id": orig_id,
            "full_review": chunk_meta["full_review"],
            "username": chunk_meta["username"],
            "score": float(score)
        })
        seen.add(orig_id)

        if len(results) >= k:
            break

    return pd.DataFrame(results)

