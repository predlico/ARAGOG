from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
import qdrant_client
from llama_index.llms.openai import OpenAI
import openai
import json
from eval_utils import load_eval_data, run_evaluations

# Load configuration settings from a JSON file.
with open('config.json') as config_file:
    config = json.load(config_file)

# Set the OpenAI API key for authentication.
openai.api_key = config['openai_api_key']

# Initialize the Qdrant client with configuration settings.
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)

# Set up the vector store and index for document storage and retrieval.
vector_store = QdrantVectorStore(client=client, collection_name="ai-arxiv")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Initialize the language model with specified parameters.
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

from llama_index.core import PromptTemplate

text_qa_template_str = (
    "You are an expert Q&A system that is trusted around the world for your factual accuracy.\n"
    "Always answer the query using the provided context information, and not prior knowledge. "
    "Ensure your answers are fact-based and accurately reflect the context provided.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' "
    "or anything along those lines.\n"
    "3. Focus on succinct answers that provide only the facts necessary, do not be verbose."
    "Your answers should be max two sentences, up to 250 characters.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
text_qa_template = PromptTemplate(text_qa_template_str)

query_engine = index.as_query_engine(llm=llm,
                                     text_qa_template=text_qa_template,
                                     similarity_top_k=5)

# Assuming all JSON files are in the same directory and have been downloaded to the local environment
file_names = [
    'questions_ground_truth_bert_v2.json',
    'questions_ground_truth_distilbert_v2.json',
    'questions_ground_truth_hellaswag_v2.json',
    'questions_ground_truth_llama_v2.json',
    'questions_ground_truth_mmlu_v2.json',
    'questions_ground_truth_red_teaming_v2.json'
]

# Load evaluation data
dir_path = 'eval_questions/'
all_data = load_eval_data(dir_path, file_names)

# Run evaluations
df = run_evaluations(all_data, llm, query_engine)
print(df)
# Save results to an Excel file
df.to_pandas().to_excel("eval_results_v3.xlsx")

#### reranker
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_api_key = config['cohere_api_key']
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)
df = run_evaluations(all_data, llm, query_engine)
# df.to_pandas().to_excel("eval_results_v4.xlsx")
print(df)

# MMR
query_engine = index.as_query_engine(vector_store_query_mode="mmr",
                                     similarity_top_k=3)

df = run_evaluations(all_data, llm, query_engine)
print(df)

# HyDE
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

df = run_evaluations(all_data, llm, query_engine = hyde_query_engine)
print(df)