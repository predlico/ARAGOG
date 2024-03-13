from datasets import load_dataset
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
from utils import load_config
import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

WINDOW_SIZE = 3

# Load configuration settings
config = load_config('resources/config.json')
os.environ['OPENAI_API_KEY'] = config['openai_api_key']
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)
# base node parser is a sentence splitter
text_splitter = SentenceSplitter()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

#load the data
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset['train'])

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
documents = [Document(text=content) for content in df['content'].head(3)]
nodes = node_parser.get_nodes_from_documents(documents)


# Initialize embedding model and Qdrant client
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)

# Setup vector store and storage context
vector_store = QdrantVectorStore(client=client, collection_name="sentence-window-full")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Define a function to divide the nodes list into chunks of n size
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Initialize a counter to keep track of the number of nodes processed
nodes_processed = 0

# Iterate over the nodes list in chunks of 2000 and update the index
for chunk in chunks(nodes, 2000):
    index = VectorStoreIndex(nodes=chunk, storage_context=storage_context)
    nodes_processed += len(chunk)
    print(f"{nodes_processed} nodes have been processed.")