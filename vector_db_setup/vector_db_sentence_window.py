from datasets import load_dataset
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
from utils import chunked_iterable, load_config
import openai

# Hardcoded values for easy adjustment
CHUNK_SIZE = 1000 #only for db upload
WINDOW_SIZE = 3

# Load configuration settings
config = load_config('config.json')
openai.api_key = config['openai_api_key']

# Initialize the sentence window node parser with hardcoded window size
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=WINDOW_SIZE,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

# Load dataset and convert to DataFrame for easier manipulation
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset['train'])

# Create Document objects and process them into nodes
documents = [Document(text=content) for content in df['content']]
nodes = node_parser.get_nodes_from_documents(documents)

# Initialize embedding model and Qdrant client
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)

# Setup vector store and storage context
vector_store = QdrantVectorStore(client=client, collection_name="ai-arxiv-sentence-window")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index document nodes in chunks with hardcoded chunk size
for chunk in chunked_iterable(nodes, CHUNK_SIZE):
    index = VectorStoreIndex(chunk, storage_context=storage_context)
    # Note: If embedding generation in batches is needed, it can be added here
