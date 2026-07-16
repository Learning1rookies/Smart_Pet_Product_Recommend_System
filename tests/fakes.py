import hashlib
import math


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        dimensions = 32
        vector = [0.0] * dimensions
        tokens = list(text.lower()) if any("\u4e00" <= char <= "\u9fff" for char in text) else text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FakeReranker:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 - index * 0.01 for index, _ in enumerate(documents)]
