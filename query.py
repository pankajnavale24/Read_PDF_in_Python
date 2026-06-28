from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Initialize the same embedder used previously
# (Crucial: Must match the model used when creating the DB)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Load the existing Vector Store
# Do NOT use .from_documents() here; just initialize with the persist_directory
vector_store = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embedding_model
)

# 3. Define your query and convert to a vector
query_text = "What is the future of india"
query_vector = embedding_model.embed_query(query_text)

# 4. Perform the similarity search using the vector
# This returns the most similar documents found in your './chroma_db'
results = vector_store.similarity_search_by_vector(query_vector, k=2)

# 5. Print the results
print(f"\n--- Search results for: '{query_text}' ---")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print(f"Metadata: {doc.metadata}")

# results = vector_store.similarity_search_by_vector_with_relevance_scores(
#     query_vector,
#     k=100
# )

# for i, (doc, score) in enumerate(results, start=1):
#     print(f"Chunk {i}:")
#     print(doc.page_content)
#     print(f"Score: {score}")
#     print("-" * 50)