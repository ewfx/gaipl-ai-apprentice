Gen AI Integrated Platform Environment: This solution is based on having a common platform for support system operations using GEN AI and machine learning.


🎥 Demo
demo is aatached as zip file under demo folder
🖼️ Screenshots:

Screenshot 1

💡 Inspiration
Last year we have submitted a white paper on similar lines where we can leverage our solution to integrate with Application/System logs , Email and Knowledge Base and  then serve as common platform to detect anomalies and suggest relevant solutions or provide chat related capabilities to end users to get the relevant support document based on the interactions.

⚙️ What It Does
Our solution does the following:
    - Processes the information (EMAILS/DOCUMENTS)
    - Passes the extracted information to NLP Engine for Entity recognition , Semantic Search , Text Embeddings and Sequence Classification
    - Vector DB for persisting the embeddings
    - Conversational AI with GPT
    - Data Validation

🛠️ How We Built It

PyMuPDF: Extracting text from PDFs while preserving structure.
Tesseract OCR: Converting scanned documents to searchable text.
LayoutLM: Analyzing document layouts (tables, headers) for context.

NLP Pipeline
BERT (Hugging Face): Fine-tuned for multi-label classification of regulations.
Sentence Transformers: Generating embeddings for semantic search.

Vector Database
FAISS: Efficient similarity search for regulatory text chunks.

Conversational AI

GPT-3.5 (OpenAI): Generating human-readable answers via LangChain.
LangChain: Orchestrating retrieval-augmented generation (RAG) workflows.

APIs & Backend
FastAPI: Building REST endpoints for document uploads and queries.
Uvicorn: ASGI server for high-performance API deployment.

🚧 Challenges We Faced
Describe the major technical or non-technical challenges your team encountered.

🏃 How to Run
Clone the repository
git clone https://github.com/ewfx/gaipl-ai-apprentice
Install dependencies
pip install -r requirements.txt (for Python)
Run the project
python main.py
Need to have OPENAI key for running
🏗️ Tech Stack
🔹 Frontend: Jinja2
🔹 Backend:  FastAPI
🔹 Database: FAISS
🔹 Other: OpenAI API / BERT / 
👥 Team
Your Name - Sagun246 | https://www.linkedin.com/in/sagun-gupta-40159329/
Teammate 2 - GitHub | LinkedIn