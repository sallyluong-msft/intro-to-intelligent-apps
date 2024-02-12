import os
from langchain.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables
if load_dotenv():
    print("Found Azure OpenAI Endpoint: " + os.getenv("AZURE_OPENAI_ENDPOINT"))
else: 
    print("No file .env found")

# Create an instance of Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    temperature = 0
)

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

data_dir = "labs/03-orchestration/02-Embeddings/data/movies"

documents = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader).load()

# Support for callbacks
from langchain.callbacks import get_openai_callback

from langchain_openai import AzureOpenAIEmbeddings

embeddings_model = AzureOpenAIEmbeddings(    
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version = os.getenv("OPENAI_EMBEDDING_API_VERSION"),
    model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
)

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
document_chunks = text_splitter.split_documents(documents)

from langchain.vectorstores import Qdrant

qdrant = Qdrant.from_documents(
    document_chunks,
    embeddings_model,
    location=":memory:",
    collection_name="movies",
)

retriever = qdrant.as_retriever()


#print(f"Total tokens used: {total_tokens}")
    
from langchain.indexes import VectorstoreIndexCreator

loader = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader)

index = VectorstoreIndexCreator(
    embedding=embeddings_model
    ).from_loaders([loader])

query = "Tell me about the latest Ant Man movie. When was it released? What is it about?"
index.query(llm=llm, question=query)

# Run the chain, using the callback to capture the number of tokens used
with get_openai_callback() as callback:
    index.query(llm=llm, question=query)
    total_tokens = callback.total_tokens

print(f"Total tokens used: {total_tokens}")