from sentence_transformers import SentenceTransformer
import requests 
import numpy as np
import pandas as pd
from vector_search_engine import VectorSearchEngine  
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch
# Import klasy


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
    }

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def hit_rate2(search_engine, ground_truth, num_results=5):
    hits = 0
    
    for item in ground_truth:
        #print(item)
        query = item['question']
        true_document_id = item['document']
        
        v_query = embedding_model.encode(query)
        results = search_engine.search(v_query, num_results=num_results)
        
        for result in results:
            if result['id'] == true_document_id:
                hits += 1
                break
    
    return hits / len(ground_truth)


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

#print(search_result)

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')


relevance_total = []

# for q in tqdm(ground_truth):
#     doc_id = q['question']
#     relevance = [d['id'] == doc_id for d in search_result]
#     relevance_total.append(relevance)

hr = hit_rate2(search_engine, ground_truth, num_results=5)
print(hr)

# print(relevance_total)
# print(hit_rate(relevance_total))
