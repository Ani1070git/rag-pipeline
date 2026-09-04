from main import split_documents
from langchain.schema import Document

def test_split_documents():
    # Create fake pages instead of loading real PDF
    fake_pages = [
        Document(page_content="Python generators are functions that use yield instead of return. They are memory efficient."),
        Document(page_content="List comprehensions are a concise way to create lists in Python using square brackets."),
        Document(page_content="Decorators in Python are functions that modify the behavior of other functions.")
    ]
    chunks = split_documents(fake_pages)
    assert len(chunks) > 0
    print(f"✅ Created {len(chunks)} chunks")

def test_chunk_size():
    fake_pages = [
        Document(page_content="A" * 2000)  # long text to force splitting
    ]
    chunks = split_documents(fake_pages)
    for chunk in chunks:
        assert len(chunk.page_content) <= 1200
    print("✅ Chunk sizes are within limit")

def test_chunk_overlap():
    fake_pages = [
        Document(page_content="B" * 3000)
    ]
    chunks = split_documents(fake_pages)
    assert len(chunks) > 1
    print("✅ Large document split into multiple chunks")