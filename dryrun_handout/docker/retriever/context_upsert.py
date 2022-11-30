# pincone mydata upsert

from datasets import load_dataset, Dataset
from sentence_transformers import models, SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import pinecone
from tqdm.auto import tqdm  # progress bar
import warnings
warnings.filterwarnings("ignore")

def context_upsert(args, preprocessed_text):
    # upload to pinecone
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    pinecone.init(api_key=args['API_KEY'], environment='us-west1-gcp')

    if args['index_name'] in pinecone.list_indexes():
        return print('Index already exists')
    
    if pinecone.list_indexes() != []:
        pinecone.delete_index(pinecone.list_indexes()[0])

    pinecone.create_index(
        name=args['index_name'], dimension=model.get_sentence_embedding_dimension(), metric='cosine'
    )

    dic = {}
    dic['id'] = [str(i) for i in range(len(preprocessed_text))]
    dic['context'] = preprocessed_text

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
    dataset = dataset.map(lambda x: {
        'encoding': model.encode(x['context']).tolist()
    }, batched=True, batch_size=4)

    # initialize connection to the new index
    index = pinecone.Index(args['index_name'])

    upserts = [(v['id'], v['encoding'], {'text': v['context']}) for v in dataset]
    # now upsert in chunks(50)
    for i in tqdm(range(0, len(upserts), 50)):
        i_end = i + 50
        if i_end > len(upserts): i_end = len(upserts)
        index.upsert(vectors=upserts[i:i_end])

    print('Pinecone upsert completed')
