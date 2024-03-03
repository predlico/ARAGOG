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
vector_store = QdrantVectorStore(client=client, collection_name="six_papers_arxiv_ai")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Initialize the language model with specified parameters.
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
query_engine = index.as_query_engine(llm=llm)

# Assuming all JSON files are in the same directory and have been downloaded to the local environment
file_names = [
    'questions_ground_truth_bert.json',
    'questions_ground_truth_distilbert.json',
    'questions_ground_truth_hellaswag.json',
    'questions_ground_truth_llama.json',
    'questions_ground_truth_mmlu.json',
    'questions_ground_truth_red_teaming.json'
]

# Load evaluation data
dir_path = 'eval_questions/'
all_data = load_eval_data(dir_path, file_names)

# Run evaluations
df = run_evaluations(all_data, llm, query_engine)

# Save results to an Excel file
df.to_excel("eval_results.xlsx")
