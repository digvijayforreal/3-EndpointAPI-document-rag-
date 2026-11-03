
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

pages = load_pdf("documents/sample.pdf")
chunks = chunk_text(pages)

print("Total pages with text:", len(pages))
print("Total chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\n--- CHUNK", chunk["chunk_id"], "---")
    print("Page:", chunk["page"])
    print(chunk["text"][:300])
