import duckdb
import numpy as np
import pandas as pd
from langchain_community.embeddings import LlamaCppEmbeddings
import pprint as pp



db_path = "website_duckdb.db"
conn = duckdb.connect(db_path)


# Configure embedding model
model_path = "bge-large-en-v1.5-f16.gguf"

embed_model = LlamaCppEmbeddings(
    model_path=model_path,
    n_ctx=512,
    n_batch=8
)


def search_similar(ask, top_k=10):
    query_embedding = embed_model.embed_query(ask)
    query_embedding_np = np.array(query_embedding)

    # Get saved embeddings
    results = conn.execute("SELECT distinct on (text) text, embedding FROM website.llamaindex_rag where len(text)>75").fetchall()

    similarities = []
    for row in results:
        db_embedding = np.frombuffer((np.array(row[1]).astype(np.float32).tobytes()), dtype=np.float32)  # Convert BLOB a array
        similarity = np.dot(query_embedding_np, db_embedding) / (np.linalg.norm(query_embedding_np) * np.linalg.norm(db_embedding))  # Calc similarity coseno
        similarities.append((row[0], similarity))

    # Sort by similarity and return `top_k` closest
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


query = "what is the first day of class fall 2025?"
results = search_similar(query)

#print(results)
pp.pprint(results)





