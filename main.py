from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import gc

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def clear_chroma_db():
    if os.path.exists("./chroma_db"):
        try:
            gc.collect()
            shutil.rmtree("./chroma_db")
        except Exception as e:
            print(f"Warning: Could not clear chroma_db: {e}")

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} Pages")
    return pages

def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = splitter.split_documents(pages)
    print(f"split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    emmbeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=emmbeddings,
        persist_directory="./chroma_db"
    )
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store

def query_rag(vector_store, question):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="groq/compound-mini",
        messages = [
            {
                "role" : "system",
                "content" : "Answer questions based only on the provided context. If the answer is not in the context, say 'I don't know'."
            },
            {
                "role" : "user",
                "content" : f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{uuid.uuid4()}.pdf"

    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    clear_chroma_db()

    pages = load_document(file_path)
    chunks = split_documents(pages)
    vector_store = create_vector_store(chunks)

    os.remove(file_path)

    return {"message": "PDF processed successfully", "chunks": len(chunks)}

@app.post("/reset")
async def reset():
    clear_chroma_db()
    return {"message": "Reset successful"}

@app.post("/query")
async def query_endpoint(request: dict):
    try:
        question = request.get("question")
        
        if not question:
            return {"error": "No question provided"}
        
        if not os.path.exists("./chroma_db"):
            return {"error": "No PDF uploaded yet. Please upload a PDF first."}
        
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        answer = query_rag(vector_store, question)
        return {"answer": answer}
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)