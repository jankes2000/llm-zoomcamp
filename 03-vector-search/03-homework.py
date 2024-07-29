from sentence_transformers import SentenceTransformer
import requests 
import numpy as np
import pandas as pd
from vector_search_engine import VectorSearchEngine  
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch


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



def hit_rate(search_engine, ground_truth, num_results=5):
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


def elasticsearch_search(es, index_name, query_vector, num_results=5):
    search_query = {
        "track_total_hits": True,
        "size": num_results,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }
    response = es.search(index=index_name, body=search_query)
    return response['hits']['hits']

def hit_rate_elasticsearch(es, index_name, embedding_model, ground_truth, num_results=5):
    hits = 0
    
    for item in tqdm(ground_truth):
        query = item['question']
        true_document_id = str(item['document'])
        
        v_query = embedding_model.encode(query).tolist()
        results = elasticsearch_search(es, index_name, v_query, num_results=num_results)
        
        for result in results:
            if result['_id'] == true_document_id:
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

# hr = hit_rate(search_engine, ground_truth, num_results=5)
# print(hr)

# print(relevance_total)
# print(hit_rate(relevance_total))

es = Elasticsearch("http://localhost:9200")

# Define the index settings and mappings
index_name = "qa_embeddings"
embedding_dim = 768  # Change this to the actual dimension of your embeddings

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "standard"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "course": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": embedding_dim
            }
        }
    }
}

# Create the index
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
es.indices.create(index=index_name, body=index_settings)

for i, doc in enumerate(documents_raw):
    embedding = embeddings[i].tolist()  # Convert embedding to list for JSON serialization
    doc_body = {
        "question": doc['question'],
        "answer": doc['text'],
        "course": doc['course'],
        "embedding": embedding
    }
    es.index(index=index_name, body=doc_body, id=doc['id'])

    # Define the query vector
query_vector = result_vector.tolist()

# Perform the search
search_query = {
    "size": 5,
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}

response = es.search(index=index_name, body=search_query)

# Get the ID of the document with the highest score
top_doc_id = response['hits']['hits'][0]['_id']
print(f"The ID of the document with the highest score is: {top_doc_id}")

results = elasticsearch_search(es, index_name, result_vector, num_results=5)
index_name = "qa_embeddings"
hr_elastic = hit_rate_elasticsearch(es, index_name, embedding_model, ground_truth, num_results=5)
print(f"Hit-rate for Elasticsearch: {hr_elastic:.2f}")