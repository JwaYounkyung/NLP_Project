# query retrieval using pinecone index

from datasets import load_dataset
from sentence_transformers import models, SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import pinecone
from tqdm.auto import tqdm  # progress bar
import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

API_KEY = '54c68dfa-f01b-4423-91f9-5c7938a30cfc'
pinecone.init(api_key=API_KEY, environment='us-west1-gcp')

# check if index already exists, if not assert
assert 'squad-index' in pinecone.list_indexes(), "Index does not exist"

# import pinecone index
index = pinecone.Index('squad-index')

query = "who added 8 more goals in 2006?"
query_encoded = model.encode([query]).tolist()
retriever_result = index.query(query_encoded, top_k=2, include_metadata=True) # id, metadata, score

print(retriever_result['matches'][0]['metadata']['text'])
