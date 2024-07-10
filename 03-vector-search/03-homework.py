from sentence_transformers import SentenceTransformer
import requests 
import numpy as np

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

documents = []


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
