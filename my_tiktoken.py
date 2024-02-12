import os
import tiktoken

encoding = tiktoken.get_encoding("p50k_base")
token = encoding.encode("Hello world, this is fun!")
print(token)

# Write code to display the text from the tokens below
decoded_text = [encoding.decode_single_token_bytes(token) for token in encoding.encode("Hello world, this is fun!")]
print(decoded_text)

def get_num_tokens_from_string(string: str, encoding_name: str='p50k_base') -> int:
    """Returns the number of tokens in a text by a given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

print(get_num_tokens_from_string("Hello World, this is fun!"))

# Open the file with information about movies
movie_data = os.path.join(os.getcwd(), "labs/03-orchestration/01-Tokens/movies.csv")
content = open(movie_data, "r", encoding="utf-8").read()

# Use tiktoken to tokenize the content and get a count of tokens used.
encoding = tiktoken.get_encoding("p50k_base")
print (f"Token count: {len(encoding.encode(content))}")