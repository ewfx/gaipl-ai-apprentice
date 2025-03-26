from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from typing import Dict

class RegulatoryQAChain:
    def __init__(self, vector_db):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        if not vector_db.vector_store:
            raise ValueError("Vector store must be initialized first")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.vector_store.as_retriever(),
            return_source_documents=True
        )

    def generate_response(self, query: str) -> Dict:
        """Generate answer with citations"""
        print("inside qa")
        result = self.qa_chain({"query": query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata["source"] for doc in result["source_documents"]]
        }