from langchain.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


import openai
import os
from dotenv import load_dotenv

# Load environment variables
if load_dotenv():
    print("Found Azure OpenAI API Base Endpoint: " + os.getenv("AZURE_OPENAI_ENDPOINT"))
else: 
    print("No file .env found")
    
# Create an instance of Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
)

# Define the prompt we want the AI to respond to - the message the Human user is asking
msg = HumanMessage(content="Explain step by step. How old is the president of USA?")

# Call the API
r = llm.invoke([msg])

# Print the response
print(r.content)

# Create a prompt template with variables, note the curly braces
prompt = PromptTemplate(
    input_variables=["input"],
    template="What interesting things can I make with a {input}?",
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
response = chain.invoke({"input": "raspberry pi"})

# As we are using a single input variable, you could also run the string like this:
# response = chain.run("raspberry pi")

print(response['text'])

