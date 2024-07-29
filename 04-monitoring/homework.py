import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rouge import Rouge

github_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv'
url = f'{github_url}?raw=1'
df = pd.read_csv(url)

df = df.iloc[:300]

embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

answer_llm = df.iloc[0].answer_llm
embedding = embedding_model.encode(answer_llm)

first_value = embedding[0]
print(first_value)

evaluations = []

for _, row in df.iterrows():
    answer_llm = row['answer_llm']
    answer_orig = row['answer_orig']
    
    embedding_llm = embedding_model.encode(answer_llm)
    embedding_orig = embedding_model.encode(answer_orig)
    
    dot_product = np.dot(embedding_llm, embedding_orig)
    evaluations.append(dot_product)

percentile_75 = np.percentile(evaluations, 75)
print(percentile_75)

def normalize(v):
    norm = np.sqrt(np.sum(v * v))
    return v / norm

cosine_scores = []

for _, row in df.iterrows():
    answer_llm = row['answer_llm']
    answer_orig = row['answer_orig']
    
    embedding_llm = embedding_model.encode(answer_llm)
    embedding_orig = embedding_model.encode(answer_orig)
    
    norm_llm = normalize(embedding_llm)
    norm_orig = normalize(embedding_orig)
    
    cosine_similarity = np.dot(norm_llm, norm_orig)
    cosine_scores.append(cosine_similarity)

cosine_percentile_75 = np.percentile(cosine_scores, 75)
print(cosine_percentile_75)

rouge_scorer = Rouge()

row = df.iloc[10]
scores = rouge_scorer.get_scores(row['answer_llm'], row['answer_orig'])[0]

rouge_1_f_score = scores['rouge-1']['f']
print(rouge_1_f_score)

rouge_2_f_score = scores['rouge-2']['f']
rouge_l_f_score = scores['rouge-l']['f']

rouge_avg_score = (rouge_1_f_score + rouge_2_f_score + rouge_l_f_score) / 3
print(rouge_avg_score)

rouge_scores = []

for _, row in df.iterrows():
    scores = rouge_scorer.get_scores(row['answer_llm'], row['answer_orig'])[0]
    
    rouge_1_f = scores['rouge-1']['f']
    rouge_2_f = scores['rouge-2']['f']
    rouge_l_f = scores['rouge-l']['f']
    
    rouge_avg = (rouge_1_f + rouge_2_f + rouge_l_f) / 3
    rouge_scores.append({
        'rouge-1': rouge_1_f,
        'rouge-2': rouge_2_f,
        'rouge-l': rouge_l_f,
        'rouge_avg': rouge_avg
    })

rouge_df = pd.DataFrame(rouge_scores)

rouge_2_mean = rouge_df['rouge-2'].mean()
print(rouge_2_mean)