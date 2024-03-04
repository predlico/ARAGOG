# Importing necessary libraries for loading datasets, data manipulation, document processing, vector storage, and embeddings.
from datasets import load_dataset
import pandas as pd
from llama_index.core import Document
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter

import openai
import json

# Load the config file
with open('config.json') as config_file:
    config = json.load(config_file)

openai.api_key = config['openai_api_key']

# Load the AI research papers dataset from the Hugging Face datasets library
dataset = load_dataset("jamescalam/ai-arxiv")

# Convert the 'train' split of the dataset into a pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Define a list of specific paper titles that you are interested in
selected_titles = [
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    "Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Measuring Massive Multitask Language Understanding",
    "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    "HellaSwag: Can a Machine Really Finish Your Sentence?"
]
# Filter the DataFrame to only include papers with the titles defined above
selected_papers = df[df['title'].isin(selected_titles)]

# Additionally, filter the DataFrame to exclude the selected titles
remaining_papers = df[~df['title'].isin(selected_titles)]

# Randomly select 50 papers from the remaining dataset
random_papers = remaining_papers.sample(n=50, random_state=42)  # Use a fixed random state for reproducibility

# Combine the selected papers with the randomly selected papers
combined_papers = pd.concat([selected_papers, random_papers])

# Initialize an empty list to store document objects
documents = []

# For each paper content in the filtered DataFrame, create a Document object and append it to the list
for content in combined_papers['content']:
    doc = Document(text=content)
    documents.append(doc)


# Initialize a parser that splits text into tokens with a specified chunk size and overlap
parser = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Use the parser to split the documents into nodes (smaller text chunks) for processing
nodes = parser.get_nodes_from_documents(documents)

# Initialize the embedding model with OpenAI's text embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Create a Qdrant client to connect to the Qdrant vector database service
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)

# Initialize the vector store to store the embeddings in Qdrant under the specified collection name
vector_store = QdrantVectorStore(client=client, collection_name="fifty_papers_arxiv_ai")

# Set up the storage context for storing vectors using the vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create an index that maps the nodes to their corresponding vector embeddings in the vector store
index = VectorStoreIndex(nodes, storage_context=storage_context)
