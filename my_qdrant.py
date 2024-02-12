import os
from dotenv import load_dotenv

# Load environment variables
if load_dotenv():
    print("Found Azure OpenAI Endpoint: " + os.getenv("AZURE_OPENAI_ENDPOINT"))
else: 
    print("No file .env found")

# Langchain document loader to load all of the movie data into memory
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='labs/03-orchestration/03-Qdrant/movies.csv', source_column='original_title', encoding='utf-8', csv_args={'delimiter':',', 'fieldnames': ['id', 'original_language', 'original_title', 'popularity', 'release_date', 'vote_average', 'vote_count', 'genre', 'overview', 'revenue', 'runtime', 'tagline']})
data = loader.load()
# data = data[1:51] # You can uncomment this line to load a smaller subset of the data if you experience issues with token limits or timeouts later in this lab.
print('Loaded %s movies' % len(data))

# Create Azure OpenAI embedding and chat completion deployments
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# Create an Embeddings Instance of Azure OpenAI
embeddings = AzureOpenAIEmbeddings(
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version = os.getenv("OPENAI_EMBEDDING_API_VERSION"),
    model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
)


# Create a Chat Completion Instance of Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
)

# Load Movies into Qdrant
from langchain.vectorstores import Qdrant

url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    data,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="my_movies",
)



from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

index_creator = VectorstoreIndexCreator(embedding=embeddings)
docsearch = index_creator.from_loaders([loader])

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question", return_source_documents=True)

query = "Do you have a column called popularity?"
response = chain.invoke({"question": query})
print(response['result'])

#print(response['source_documents'])

query = "If the popularity score is defined as a higher value being a more popular movie, what is the name of the most popular movie in the data provided?"
response = chain.invoke({"question": query})
print(response['result'])
