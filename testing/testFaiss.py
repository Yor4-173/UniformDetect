import faiss
import numpy as np
from src.embedding_app import load_model, get_embedding

model = load_model()

# Gallery embeddings
gallery_paths = [
    "dataset/valid/female_uniform/0012.jpg",
    "dataset/valid/male_uniform/0009.jpg",
]
gallery_embeddings = [get_embedding(p, model) for p in gallery_paths]
gallery_embeddings = np.array(gallery_embeddings).astype("float32")

# Create FAISS index
d = gallery_embeddings.shape[1]  # dimension
index = faiss.IndexFlatL2(d)
index.add(gallery_embeddings)

# Query 
query_path = "testing/TestImg3.png"
query_embedding = get_embedding(query_path, model).astype("float32").reshape(1, -1)

D, I = index.search(query_embedding, k=1)

print("Nearest distance:", D[0][0])
print("Matched index:", I[0][0])

# hreshold
threshold = 0.5
if D[0][0] < threshold:
    print("Correct Uniform")
else:
    print("Wrong Uniform")
