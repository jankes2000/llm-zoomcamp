from sentence_transformers import SentenceTransformer
import requests 
import numpy as np
from vector_search_engine import VectorSearchEngine  # Import klasy


# class VectorSearchEngine():
#     def __init__(self, documents, embeddings):
#         self.documents = documents
#         self.embeddings = embeddings

#     def search(self, v_query, num_results=10):
#         scores = self.embeddings.dot(v_query)
#         idx = np.argsort(-scores)[:num_results]
#         return [self.documents[i] for i in idx]


model_name = "multi-qa-distilbert-cos-v1"
embedding_model = SentenceTransformer(model_name)

user_question = "I just discovered the course. Can I still join it?"

result_vector = embedding_model.encode(user_question)

print(result_vector[0])


base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
doc_raw_before_filtering = docs_response.json()

#documents = []


documents_raw = [doc for doc in doc_raw_before_filtering if doc['course'] == 'machine-learning-zoomcamp']

print("filtered_document_count = ", len(documents_raw))
embeddings = []
for doc in documents_raw:
    question = doc['question']
    answer = doc['text']
    qa_text = f'{question} {answer}'
    embedding = embedding_model.encode(qa_text)
    embeddings.append(embedding)

X = np.array(embeddings)

print(X.shape)

v = result_vector
scores = X.dot(v)

highest_score = np.max(scores)
print(highest_score)

search_engine = VectorSearchEngine(documents=documents_raw, embeddings=X)
search_result = search_engine.search(v, num_results=5)

print(search_result)