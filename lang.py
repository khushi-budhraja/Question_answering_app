from langchain.document_loaders import PyPDFLoader
import redis
import os
from langchain.vectorstores.redis import Redis
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import openai

openai.api_key = os.environ['OPEN_API_KEY']

openai_api_key=os.environ['OPEN_API_KEY']
load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)

redis_instance = redis.Redis(
        host = "127.0.0.1",
        port = "6379"
    )

def vectorize(pdf,index):
    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()
    i= 0
    for page in pages:
        print(i)
        i+=1
        #create index
        Redis.from_documents(
            [page],
            embeddings,
            redis_url = "redis://127.0.0.1:6379",
            index_name = index
        )
    document_store = Redis.from_existing_index(
            index_name = index,
            redis_url = "redis://127.0.0.1:6379",
            embedding = OpenAIEmbeddings(openai_api_key = openai_api_key)
        )
    return document_store

def load_data(index,pdf):
    try:
        #to load
        document_store = Redis.from_existing_index(
            index_name = index,
            redis_url = "redis://127.0.0.1:6379",
            embedding = OpenAIEmbeddings(openai_api_key = openai_api_key)
        )
        print(f"Data loaded from existing index {index}")
    except:
        #to insert 
        document_store = vectorize(pdf,index)
        print(f"Data inserted for index {index}")
    return document_store

pdf = "covid.pdf"
data_store = load_data("test",pdf)

ques = "Germay population"

#similarity search
related_doc = data_store.similarity_search(ques,k=3)

content = [ doc.page_content for doc in related_doc]
knowledge = "\n".join(content)
query =  "Based on the below knowledge answer this question." + ques + "\n" + knowledge

chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
              "role": "system",
              "content": "You are an incredible AI who can answer any question related to covid in an unique way."
            },
            {
                "role": "user",
                "content" : query
            }
        ]
    )

ans = chat["choices"][0]["message"]["content"]
print(ans)



'''
create environment : python -m venv enc
activate environment : enc\Scripts\activate
pip install pypdf
pip install redis
pip install langchain
pip install python-dotenv
pip install tiktoken
pip install openai
'''
