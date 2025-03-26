from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict

import os

os.environ["OPENAI_API_KEY"] = ""

class RegulatoryVectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(["apple", "banana"], self.embeddings)

    def create_vector_db(self, documents: List[Dict]):
        """Create FAISS vector database from documents"""
        texts = [doc["text"] for doc in documents]
        metadatas = [{"source": doc["source"]} for doc in documents]

        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)