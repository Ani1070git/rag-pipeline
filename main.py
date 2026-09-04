from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

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

if __name__ == "__main__":
    import os
    
    if os.path.exists("./chroma_db"):
        print("Loading existing vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        print("Vector store loaded!")
    else:
        print("Creating new vector store...")
        pages = load_document("sample.pdf")
        chunks = split_documents(pages)
        vector_store = create_vector_store(chunks)
    
    question = "What are the best practices for using dictionaries?"
    answer = query_rag(vector_store, question)
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {answer}")