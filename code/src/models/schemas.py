from pydantic import BaseModel

class DocumentUpload(BaseModel):
    file_path: str
