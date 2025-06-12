import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import uvicorn
# Close any existing Chroma client cleanly before deleting
if retriever is not None:
    try:
        vectordb._persist_directory = None  # prevents it from persisting again
        vectordb = None
        retriever = None
        qa_chain = None
    except Exception as e:
        print("Error cleaning up:", e)

# Create FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = "./chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Load HF model for QA
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

retriever = None
qa_chain = None

@app.get("/")
def main():
    content = """
    <html>
        <body>
        <h2>Upload a PDF</h2>
        <form action="/upload/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
        </form>
        <h2>Ask a Question</h2>
        <form action="/ask/" method="get">
            <input name="query" type="text">
            <input type="submit">
        </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/upload/")
def upload_pdf(file: UploadFile = File(...)):
    global retriever, qa_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file.filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
        vectordb.persist()

        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return {"message": "PDF uploaded and indexed successfully."}

@app.get("/ask/")
def ask_query(query: str):
    global qa_chain
    if qa_chain is None:
        return {"error": "Please upload a PDF first."}

    response = qa_chain.run(query)
    return {"query": query, "answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)