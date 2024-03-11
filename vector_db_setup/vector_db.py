# Importing necessary libraries for loading datasets, data manipulation, document processing, vector storage, and embeddings.
from datasets import load_dataset
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from utils import chunked_iterable, load_config
import openai

# Hardcoded values for easy adjustment
CHUNK_SIZE = 1000 #only for db upload
TOKEN_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Load the config file
config = load_config('resources/config.json')
openai.api_key = config['openai_api_key']

# Load dataset and convert to DataFrame for easier manipulation
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset['train'])

# Prepare document objects from the dataset for indexing
documents = [Document(text=content) for content in df['content']]

# Initialize a text splitter with hardcoded values for chunking documents
parser = TokenTextSplitter(chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(documents)

# Setup the embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Setup Qdrant client and vector store for storing embeddings
client = qdrant_client.QdrantClient(url=config['qdrant_url'], api_key=config['qdrant_api_key'])
vector_store = QdrantVectorStore(client=client, collection_name="ai-arxiv")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Process and index document nodes in batches with hardcoded chunk size
for chunk in chunked_iterable(nodes, CHUNK_SIZE):
    index = VectorStoreIndex(chunk, storage_context=storage_context)
    # Note: Embedding generation in batches could be added here if necessary
