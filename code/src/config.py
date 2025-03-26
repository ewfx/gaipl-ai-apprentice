import os

class Config:
    CHUNK_SIZE = 512
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    os.environ["OPENAI_API_KEY"] = ""