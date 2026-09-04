from main import load_document, split_documents

def test_load_document():
    pages = load_document("sample.pdf")
    assert len(pages) > 0
    print(f"✅ Loaded {len(pages)} pages")

def test_split_documents():
    pages = load_document("sample.pdf")
    chunks = split_documents(pages)
    assert len(chunks) > len(pages)
    assert len(chunks) > 0
    print(f"✅ Created {len(chunks)} chunks")

def test_chunk_size():
    pages = load_document("sample.pdf")
    chunks = split_documents(pages)
    for chunk in chunks[:5]:
        assert len(chunk.page_content) <= 1200
    print("✅ Chunk sizes are within limit")