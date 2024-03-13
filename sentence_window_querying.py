from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from utils import load_config
from llama_index.core import Settings
import os
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

config = load_config('resources/config.json')
os.environ['OPENAI_API_KEY'] = config['openai_api_key']
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

Settings.llm = llm
Settings.embed_model = embed_model
vector_store = QdrantVectorStore(client=client, collection_name="sentence-window-full")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


query_engine = index.as_query_engine(
    similarity_top_k=3,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
window_response = query_engine.query(
    "What were the results of the description generator task"
)
print(window_response)

window = window_response.source_nodes[0].node.metadata["window"]
sentence = window_response.source_nodes[0].node.metadata["original_text"]

print('returned sentence =', sentence)
print('returned window =', window)