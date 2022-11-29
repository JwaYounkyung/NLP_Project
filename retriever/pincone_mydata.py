# pincone mydata upsert

from datasets import load_dataset
from sentence_transformers import models, SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import pinecone
from tqdm.auto import tqdm  # progress bar
import warnings
warnings.filterwarnings("ignore")

squad_dev = load_dataset('squad_v2', split='validation', streaming=True)

model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

unique_contexts = []
unique_ids = []
# make list of IDs that represent only first instance of
# each context
for row in squad_dev:
    if row['context'] not in unique_contexts:
        unique_contexts.append(row['context'])
        unique_ids.append(row['id'])

# now filter out any samples that aren't included in unique IDs
squad_dev = squad_dev.filter(lambda x: True if x['id'] in unique_ids else False)

# now encode the unique contexts
squad_dev = squad_dev.map(lambda x: {
    'encoding': model.encode(x['context']).tolist()
}, batched=True, batch_size=4)

API_KEY = '54c68dfa-f01b-4423-91f9-5c7938a30cfc'

pinecone.init(api_key=API_KEY, environment='us-west1-gcp')
# check if index already exists, if not we create it
if 'squad-index' not in pinecone.list_indexes():
    pinecone.create_index(
        name='squad-index', dimension=model.get_sentence_embedding_dimension(), metric='cosine'
    )

# initialize connection to the new index
index = pinecone.Index('squad-index')

upserts = [(v['id'], v['encoding'], {'text': v['context']}) for v in squad_dev]
# now upsert in chunks(50)
for i in tqdm(range(0, len(upserts), 50)):
    i_end = i + 50
    if i_end > len(upserts): i_end = len(upserts)
    index.upsert(vectors=upserts[i:i_end])
