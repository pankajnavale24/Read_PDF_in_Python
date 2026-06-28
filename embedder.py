from splitter import splits
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Initialize Embedder
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Create the Vector Store
vector_store = Chroma.from_documents(
    documents=splits, 
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

# 3. Access the underlying Chroma collection
# This is where the vectors are actually stored
collection = vector_store.get(include=['embeddings', 'documents', 'metadatas'])

# 4. Print the vector for the first chunk
# The 'embeddings' key contains a list of lists (each sub-list is one vector)
all_vectors = collection['embeddings']

print(f"Total number of vectors found: {len(all_vectors)}")
print(f"\nPrinting the first 5 numbers of the FIRST vector:")
print(all_vectors[0][:5]) # Printing only the first 5 to avoid a messy terminal

print(f"\nPrinting the full list of numbers for the first vector:")
print(all_vectors[0])