# pincone mydata upsert

from datasets import load_dataset, Dataset
from sentence_transformers import models, SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import pinecone
from tqdm.auto import tqdm  # progress bar
import warnings
warnings.filterwarnings("ignore")

args = {
    'index_name': 'mydata',
    'API_KEY': '54c68dfa-f01b-4423-91f9-5c7938a30cfc',
    'data_dic': 'dryrun_handout/documents/chinese_dynasties/Han_dynasty.txt',
}

dataset = load_dataset('text', data_files=[args['data_dic']])['train']

dic = {}
dic['id'] = [str(i) for i in range(len(dataset))]
dic['context'] = [v['text'] for v in dataset]

dataset = Dataset.from_dict(dic)

# make list of IDs that represent only first instance of each context
unique_contexts = []
unique_ids = []
for row in dataset:
    if row['context'] not in unique_contexts:
        unique_contexts.append(row['context'])
        unique_ids.append(row['id'])

# filter out any samples that aren't included in unique IDs
dataset = dataset.filter(lambda x: True if x['id'] in unique_ids else False)

# encode contexts
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
dataset = dataset.map(lambda x: {
    'encoding': model.encode(x['context']).tolist()
}, batched=True, batch_size=4)

# upload to pinecone
pinecone.init(api_key=args['API_KEY'], environment='us-west1-gcp')
if args['index_name'] in pinecone.list_indexes():
    pinecone.delete_index(args['index_name'])

pinecone.create_index(
    name=args['index_name'], dimension=model.get_sentence_embedding_dimension(), metric='cosine'
)

# initialize connection to the new index
index = pinecone.Index(args['index_name'])

upserts = [(v['id'], v['encoding'], {'text': v['context']}) for v in dataset]
# now upsert in chunks(50)
for i in tqdm(range(0, len(upserts), 50)):
    i_end = i + 50
    if i_end > len(upserts): i_end = len(upserts)
    index.upsert(vectors=upserts[i:i_end])
