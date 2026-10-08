from app.src.ragging import Chunker,Qdrantcollection

with open("data/bupa cover.txt", "r") as f:
    text = f.read()
chunks = Chunker().chunk_text(text)
built_chunks = Chunker.build_chunks(list_of_texts=chunks, document_name="bupa cover")
qdrant_collection = Qdrantcollection(collection_name='bupa cover')
qdrant_collection.upsert_chunks(chunks=built_chunks)