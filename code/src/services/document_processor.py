import fitz  # PyMuPDF
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF documents"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            raise RuntimeError(f"Text extraction failed: {str(e)}")

    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """Split document into contextual chunks with validation"""
        if text is None or not isinstance(text, str) or not text.strip():
            return []

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in sentences:
            sent_length = len(sent.split())
            if current_length + sent_length > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                try:
                    embedding = self.sentence_model.encode(chunk_text)
                    chunks.append({
                        "text": chunk_text,
                        "embedding": embedding
                    })
                except Exception as e:
                    print(f"Embedding generation failed: {e}")
                current_chunk = []
                current_length = 0
            current_chunk.append(sent)
            current_length += sent_length

        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            try:
                embedding = self.sentence_model.encode(chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"Final embedding generation failed: {e}")

        return chunks