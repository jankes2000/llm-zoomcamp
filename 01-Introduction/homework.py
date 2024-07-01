import json
import requests 
import tiktoken
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm



def elastic_search(query, result_count):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                }
                ,
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits'][:result_count]:
        result_docs.append(hit['_source'])
    
    return result_docs

def rag_only_prompt(query, result_count):
    search_results = elastic_search(query, result_count)
    prompt = build_prompt(query, search_results)
    #answer = llm(prompt)


    return prompt

def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"Q: {doc['question']}\nA: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

documents[0]


es_client = Elasticsearch('http://localhost:9200') 

# index_settings = {
#     "settings": {
#         "number_of_shards": 1,
#         "number_of_replicas": 0
#     },
#     "mappings": {
#         "properties": {
#             "text": {"type": "text"},
#             "section": {"type": "text"},
#             "question": {"type": "text"},
#             "course": {"type": "keyword"} 
#         }
#     }
# }

index_name = "course-questions"

# es_client.indices.create(index=index_name, body=index_settings)

# for doc in tqdm(documents):
es_client.index(index=index_name, document=doc)

query = 'How do I execute a command in a running docker container?'

# results = elastic_search(query)

# #print(json.dumps(results))

# #print(results)

# for result in results:
#     print()
#     print(result)
prompt = rag_only_prompt(query, 3)

#print(len(prompt))
#print()
#print(prompt)

encoding = tiktoken.encoding_for_model("gpt-4o")

encoded_prompt = encoding.encode(prompt)
print(encoded_prompt)
print(len(encoded_prompt))
print(encoding.decode_single_token_bytes(63842))