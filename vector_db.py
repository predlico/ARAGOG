# Importing necessary libraries for loading datasets, data manipulation, document processing, vector storage, and embeddings.
from datasets import load_dataset
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
import chromadb
from llama_index.core.node_parser import TokenTextSplitter
from utils import chunked_iterable, load_config
from llama_index.vector_stores.chroma import ChromaVectorStore
import openai
import os

# Hardcoded values for easy adjustment
CHUNK_SIZE = 1000 #only for db upload
TOKEN_CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Load the config file
load_config()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset and convert to DataFrame for easier manipulation
dataset = load_dataset("jamescalam/ai-arxiv")
df = pd.DataFrame(dataset['train'])

# # Specify the titles of the required papers
# required_paper_titles = [
#     'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
#     'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter',
#     'HellaSwag: Can a Machine Really Finish Your Sentence?',
#     'LLaMA: Open and Efficient Foundation Language Models',
#     'Measuring Massive Multitask Language Understanding',
#     'CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks',
#     'Task2Vec: Task Embedding for Meta-Learning',
#     'GLM-130B: An Open Bilingual Pre-trained Model',
#     'SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems',
#     "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
#     "PAL: Program-aided Language Models",
#     "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
#     "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
# ]
# # Filter the DataFrame to include only the required papers
# required_papers = df[df['title'].isin(required_paper_titles)]
#
# # Exclude the already selected papers to avoid duplicates and randomly sample ~40-50 papers
# remaining_papers = df[~df['title'].isin(required_paper_titles)].sample(n=40, random_state=123)
#
# # Concatenate the two DataFrames
# final_df = pd.concat([required_papers, remaining_papers], ignore_index=True)

# Prepare document objects from the dataset for indexing
documents = [Document(text=content) for content in df['content']]

# Setup the embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-large")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Classic vector DB
# Initialize a text splitter with hardcoded values for chunking documents
parser = TokenTextSplitter(chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = parser.get_nodes_from_documents(documents)

chroma_collection = chroma_client.create_collection("ai_arxiv_full")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes, storage_context=storage_context,
    embed_model=embed_model,
    use_async=True
)

# Sentence window
node_parser_sentence_window = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
nodes_sentence_window = node_parser_sentence_window.get_nodes_from_documents(documents)
nodes_sentence_window_filtered = filter_large_nodes(nodes_sentence_window)

chroma_collection_sentence_window = chroma_client.create_collection("ai_arxiv_sentence_window")

vector_store_sentence_window = ChromaVectorStore(chroma_collection=chroma_collection_sentence_window)

storage_context_sentence_window = StorageContext.from_defaults(vector_store=vector_store_sentence_window)

# Removed nodes longer than 8K, they were causing issues and these are only tables and references which are not relevant,
# in total 17 chunks or something
index = VectorStoreIndex(
    nodes_sentence_window_filtered,
    storage_context=storage_context_sentence_window,
    embed_model=embed_model,
    use_async=True,
    show_progress=True
)

# Document summary index
# LLM (gpt-3.5-turbo)
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import DocumentSummaryIndex, get_response_synthesizer

parser = TokenTextSplitter(chunk_size=3072, chunk_overlap=100)
splits_doc_summary = parser.get_nodes_from_documents(documents)

docs_for_summary = [Document(text=node.text, metadata=node.metadata) for node in splits_doc_summary]

from llama_index.llms.mistralai import MistralAI

# To customize your API key, do this
# llm = MistralAI(api_key="")
# feel free to use OpenAI models, we kept getting some weird errors.

# Initialize the sentence splitter
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)

# Initialize the response synthesizer in 'tree_summarize' mode
response_synthesizer = get_response_synthesizer(
    llm= llm,response_mode="tree_summarize", use_async=True
)

# Create the document summary index
doc_summary_index = DocumentSummaryIndex.from_documents(
    docs_for_summary,
    llm=llm,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    embed_model=embed_model,
    show_progress=True,
    use_async=True
)

doc_summary_index.storage_context.persist("ai_arxiv_doc_summary")