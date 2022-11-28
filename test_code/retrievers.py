# pip install datasets
# pip install -U sentence-transformers
# pip install pinecone-client
from datasets import load_dataset
from sentence_transformers import InputExample
from sentence_transformers import datasets
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm.auto import tqdm
import pandas as pd
import pinecone
from tqdm.auto import tqdm  # progress bar
import warnings
warnings.filterwarnings("ignore")


squad = load_dataset('squad_v2', split='train', streaming=True)
train = []
for row in tqdm(squad):
    train.append(InputExample(
        texts=[row['question'], row['context']]
    ))

# MNR loss can't have duplicate data(context, question)
batch_size = 8
loader = datasets.NoDuplicatesDataLoader(
    train, batch_size=batch_size
)

# Load pre-trained model
bert = models.Transformer('microsoft/mpnet-base')
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[bert, pooler])
loss = losses.MultipleNegativesRankingLoss(model)

epochs = 1
warmup_steps = int(len(loader) * epochs * 0.1)

# model.fit(
#     train_objectives=[(loader, loss)],
#     epochs=epochs,
#     warmup_steps=warmup_steps,
#     output_path='mpnet-mnr-squad2',
#     show_progress_bar=True
# )
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

squad_dev = load_dataset('squad_v2', split='validation', streaming=True)

squad_df = pd.DataFrame()
for row in tqdm(squad_dev):
    squad_df = squad_df.append({
        'question': row['question'],
        'context': row['context'],
        'id': row['id']
    }, ignore_index=True)


# duplicate contexts should not assigned different IDs
no_dupe = squad_df.drop_duplicates(
    subset='context',
    keep='first'
)
# drop question column
no_dupe = no_dupe.drop(columns=['question'])
# give each context a slightly unique ID
no_dupe['id'] = no_dupe['id'] + 'con'
squad_df = squad_df.merge(no_dupe, how='inner', on='context')

# make query and context id dictionary
ir_queries = {
    row['id_x']: row['question'] for i, row in squad_df.iterrows()
}
ir_corpus = {
    row['id_y']: row['context'] for i, row in squad_df.iterrows()
}

ir_relevant_docs = {key: [] for key in squad_df['id_x'].unique()}
for i, row in squad_df.iterrows():
    # we append in the case of a question ID being connected to
    # multiple context IDs
    ir_relevant_docs[row['id_x']].append(row['id_y'])
# this must be in format {question_id: {set of context_ids}}
ir_relevant_docs = {key: set(values) for key, values in ir_relevant_docs.items()}

ir_eval = InformationRetrievalEvaluator(
    ir_queries, ir_corpus, ir_relevant_docs
)
ir_eval(model)
qa = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
ir_eval(qa)

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

# and now encode the unique contexts
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

# we use this to get required index dims
model.get_sentence_embedding_dimension()

# initialize connection to the new index
index = pinecone.Index('squad-index')

upserts = [(v['id'], v['encoding'], {'text': v['context']}) for v in squad_dev]
# now upsert in chunks
for i in tqdm(range(0, len(upserts), 50)):
    i_end = i + 50
    if i_end > len(upserts): i_end = len(upserts)
    index.upsert(vectors=upserts[i:i_end])

query = "When were the Normans in Normandy?"
xq = model.encode([query]).tolist()
xc = index.query(xq, top_k=2, include_metadata=True)

xq = model.encode([
    "How many outputs are expected for each input in a function problem?"
]).tolist()
index.query(xq, top_k=5, include_metadata=True)

xq = model.encode([
    "Who used Islamic, Lombard, etc construction techniques in the Mediterranean?"
]).tolist()
index.query(xq, top_k=5, include_metadata=True)