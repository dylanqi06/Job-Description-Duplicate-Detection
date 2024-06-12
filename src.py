import pandas as pd
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

data = pd.read_csv('jobpostings.csv')
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

data['Cleaned Job Description'] = data['Job Description'].astype(str).apply(clean_html)
cleaned_jd = data['Cleaned Job Description'].tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(cleaned_jd, show_progress_bar=True)
# np.save('jd_emd.npy', embeddings)

# data['Embeddings'] = embeddings.tolist()
# data.to_csv('jd_emd.csv', index=False)

grpc_options = [
    ("grpc.max_send_message_length", 500 * 1024 * 1024),  # 500MB
    ("grpc.max_receive_message_length", 500 * 1024 * 1024),  # 500MB
]
connections.connect("default", host="localhost", port="19530", options=grpc_options)

# Create schema
fields = [
    FieldSchema(name="job_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="Job Description Embeddings")
collection = Collection("job_postings_v5", schema)

# embeddings = np.load('jd_emd.npy')
entities = [embeddings.tolist()]
insert_result = collection.insert(entities)
assigned_ids = insert_result.primary_keys

index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
collection.create_index(field_name="embedding", index_params=index_params)

collection.load()

search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

# Function to perform batch search
results = []
def batch_search(embeddings, batch_size):
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        batch_embeddings = embeddings[start:end]
        batch_results = collection.search(batch_embeddings, "embedding", search_params, limit=2)
        results.extend(batch_results)

batch_size = 16000
batch_search(embeddings, batch_size)
similarity_scores = [match.distance for result in results for match in result]

plt.hist(similarity_scores, bins=50)
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Similarity Scores')
plt.show()

threshold = 0.05

duplicate_pairs = set()
for query_index, hits in enumerate(results):
    query_id = assigned_ids[query_index]  
    for hit in hits:
        if hit.distance < threshold and hit.id != query_id:  # Ensure we don't compare the embedding with itself
            pair = tuple(sorted((query_id, hit.id)))
            duplicate_pairs.add(pair)

print(f"Number of unique duplicate pairs: {len(duplicate_pairs)}")

connections.disconnect("default")