"""One shared MiniLM encoder, loaded once and reused (loading takes a few seconds)."""
_encoder = None


def embed(texts):
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder.encode(list(texts), normalize_embeddings=True)