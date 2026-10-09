from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
import pandas as pd
from qdrant_client.http import models
from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from pathlib import Path

def to_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))

class Chunker:
    def __init__(self, embedding_model:str = "all-MiniLM-L6-v2",threshold:float = 0.99):
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.threshold = threshold
    def sentence_splitter(self,text:str):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=100,      # Maximum tokens per chunk
        chunk_overlap=0      # Overlap tokens between chunks if needed
            )
        return text_splitter.split_text(text)
    def cosine_similarity(self, vec1, vec2):
        dot_product = vec1.dot(vec2)
        norm_vec1 = (vec1 ** 2).sum() ** 0.5
        norm_vec2 = (vec2 ** 2).sum() ** 0.5
        return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 and norm_vec2 else 0.0
    def chunk_text(self, text:str):
        sentences = self.sentence_splitter(text)
        embeddings = self.model.encode(sentences,show_progress_bar=False)
        chunks = []
        current_chunk = []
        current_embedding = None
        for sentence, embedding in zip(sentences, embeddings):
            if current_embedding is None:
                current_chunk.append(sentence)
                current_embedding = embedding
            else:
                similarity = self.cosine_similarity(current_embedding, embedding)
                if similarity >= self.threshold:
                    current_chunk.append(sentence)
                    current_embedding = (current_embedding + embedding) / 2
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_embedding = embedding
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    @staticmethod
    def build_chunks(list_of_texts:list,document_name:str)->pd.DataFrame:
        all_chunks = []
        for index,text in enumerate(list_of_texts):
            all_chunks.append({
                "document": document_name,
                "index": index,
                "chunk": text,
                "id": to_uuid(str(document_name) + "_" + str(index))
            })
        return pd.DataFrame(all_chunks)

class Qdrantcollection:
    def __init__(self, path:str = "qdrant", collection_name:str = None,embedding_model:str = "all-MiniLM-L6-v2",sparse_vector_model:str = "Qdrant/bm25"):
        DATA_DIR = Path("app/data")
        self.path = DATA_DIR / path
        if collection_name is None:
            raise ValueError("collection_name can't be None")
        else:    
            self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.bm25_model = SparseTextEmbedding(sparse_vector_model)
        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Inicializa Qdrant local.
        self.client = QdrantClient(
            path=str(self.path)
        )

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.encoder.get_embedding_dimension(),
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
    def upsert_chunks(self, chunks:pd.DataFrame):
        sparse_vecs = list(self.bm25_model.embed(chunks['chunk'].tolist()))
        embeddings = self.encoder.encode(chunks['chunk'].tolist(),show_progress_bar=False)
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks['chunk'], embeddings)):
            point = models.PointStruct(
                id=chunks['id'].iloc[i],
                vector={
                "dense": vector.tolist(),
                "bm25": models.SparseVector(
                    indices=sparse_vecs[i].indices.tolist(),
                    values=sparse_vecs[i].values.tolist(),
                ),
            },
                payload={
                    "text": chunk,
                    "document": chunks['document'].iloc[i],
                    "index": int(chunks['index'].iloc[i])
                }
            )
            points.append(point)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    def search_query(self, query: str, top_k: int = 5, prefetch_k: int = 50):
        dense_q = self.encoder.encode([query], show_progress_bar=False)[0]
        sparse_q = next(self.bm25_model.query_embed(query))

        query_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_q.tolist(),
                    using="dense",
                    limit=prefetch_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_q.indices.tolist(),
                        values=sparse_q.values.tolist(),
                    ),
                    using="bm25",
                    limit=prefetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [point.payload for point in query_result.points]
    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
