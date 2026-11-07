
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = "db/chroma_db_hf"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Accept any question from the user
query = input("Ask your question: ")

retriever = db.as_retriever(
    search_kwargs={"k": 3},
    score_threshold=0.3
)

relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")
print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata.get('source')}")
    print(doc.page_content)

print("\n--- Database Diagnostic ---")

print("Total stored chunks:", db._collection.count())

matches = db.get(
    where_document={"$contains": "Omelek"},
    include=["documents"]
)

print("Chunks mentioning Omelek:", len(matches["ids"]))


print("\n--- Checking Retrieval Ranking ---")

results = db.similarity_search_with_score(
    query,
    k=50
)

for rank, (doc, distance) in enumerate(results, 1):
    if "omelek" in doc.page_content.lower():
        print(f"Found at rank: {rank}")
        print(f"Distance score: {distance:.4f}")
        print(f"Content: {doc.page_content}")
        break
else:
    print("Omelek not found in top 50 results.")
