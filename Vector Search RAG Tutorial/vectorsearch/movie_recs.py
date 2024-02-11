import pymongo
import requests
import os
from dotenv import load_dotenv

load_dotenv() # This reads the environment variables inside .env

mongodb_password = os.getenv('MONGODB_PASSWORD')
# print (mongodb_password)

mongodb_moviedb_uri = "mongodb+srv://vdalal:" + mongodb_password + "@cluster0.ojkrxkw.mongodb.net/?retryWrites=true&w=majority"
# print (mongodb_moviedb_uri)

client = pymongo.MongoClient(mongodb_moviedb_uri)
db = client.sample_mflix
collection = db.movies

# print (collection.find().limit(5))
# items = collection.find().limit(5)

# for item in items:
#    print(item)

hf_token = os.getenv('HF_TOKEN')
# print (hf_token)
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

for doc in collection.find({'plot': {'$exists': True}}).limit(1150):
    if doc.get('plot_embedding_hf') is None:
        doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
        collection.replace_one({'_id': doc['_id']}, doc)

query = "imaginary characters from outer space at war"
# query = "spy movies with lots of action for kids"

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