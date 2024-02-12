import os
import openai
from dotenv import load_dotenv
from openai import AzureOpenAI

# Open the file with information about movies
movie_data = os.path.join(os.getcwd(), "labs/03-orchestration/01-Tokens/movies.csv")
content = open(movie_data, "r", encoding="utf-8").read()

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("OPENAI_API_VERSION")
)

query = "What's the highest rated movie from the following list\n"
query += "CSV list of movies:\n"
query += content

print (f"{query[0:3000]} ...[{len(query)} characters]")

response = client.chat.completions.create(
    model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    messages = [{"role" : "assistant", "content" : query}],
)

print (response)
