from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

from multipart import file_path

from services.bert_classifier import BERTClassifier
from services.document_processor import DocumentProcessor
from services.qa_chain import RegulatoryQAChain
from services.vector_db import RegulatoryVectorDB

app = FastAPI()
processor = DocumentProcessor()
classifier = BERTClassifier()
vector_db = RegulatoryVectorDB()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-document")
async def upload_document(request: Request, file: UploadFile = File(...)):
    try:
        # Ensure upload directory exists
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()  # Use async read
            f.write(content)

        # Process document
        text = processor.extract_text(file_path)
        chunks = processor.chunk_document(text)

        # Classify and store
        classified_chunks = []
        for chunk in chunks:
            classification = classifier.predict(chunk["text"])
            classified_chunks.append({
                "text": chunk["text"],
                "source": file.filename,
                "metadata": classification
            })

        # Initialize vector DB and QA chain
        vector_db.create_vector_db(classified_chunks)
        global qa_chain
        qa_chain = RegulatoryQAChain(vector_db)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "upload_status": f"Processed {file.filename} successfully! ({len(chunks)} chunks)"
        })

    except Exception as e:
        # Clean up file if error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )



@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    try:
        print("inside ask")
        response = qa_chain.generate_response(question)
        print(response)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "answer": response,
            "question": question
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)