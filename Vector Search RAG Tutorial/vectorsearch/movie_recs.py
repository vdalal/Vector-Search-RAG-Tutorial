import pymongo
import requests
import os
from dotenv import load_dotenv

load_dotenv() # This reads the environment variables inside .env

mongodb_password = os.getenv('MONGODB_PASSWORD')
mongodb_moviedb_uri = "mongodb+srv://vdalal:" + mongodb_password + "@cluster0.ojkrxkw.mongodb.net/?retryWrites=true&w=majority"

client = pymongo.MongoClient(mongodb_moviedb_uri)
db = client.sample_mflix
collection = db.movies

hf_token = os.getenv('HF_TOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text})

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    return response.json()

# print(generate_embedding("The Matrix"))
def generate_all_movie_embeddings(lim: int):
    cnt = collection.count_documents({'plot': {'$exists': True}})
    print("Count of documents with plot summary = " + str(cnt))
    
    for doc in collection.find({'plot': {'$exists': True}}).limit(lim):
        if doc.get('plot_embedding_hf') is None:
            doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
            collection.replace_one({'_id': doc['_id']}, doc)


generate_all_movie_embeddings(7400)

query = "imaginary characters from outer space at war"
# query = "a secret agent movie with action and intrigue which is suitable for kids"
# query = "a movie about the secret service protecting the US President"

results = collection.aggregate([
    {"$vectorSearch": {
     "queryVector": generate_embedding(query),
     "path": "plot_embedding_hf",
     "numCandidates": 100,
     "limit": 4,
     "index": "PlotSemanticSearch",
    }}
]);

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')
