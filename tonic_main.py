from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, PromptTemplate
import qdrant_client
from llama_index.llms.openai import OpenAI
import openai
import json
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import AnswerSimilarityMetric, RetrievalPrecisionMetric
from utils import make_get_llama_response, remove_nul_chars_from_run_data
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

### SETUP --------------------------------------------------------------------------------------------------------------
# Load configuration settings from a JSON file.
with open('config.json') as config_file:
    config = json.load(config_file)

# Set the OpenAI API key for authentication.
openai.api_key = config['openai_api_key']
tonic_validate_api_key = config['tonic_validate_api_key']
tonic_validate_project_key = config['tonic_validate_project_key']
tonic_validate_benchmark_key = config['tonic_validate_benchmark_key']

validate_api = ValidateApi(tonic_validate_api_key)

# Initialize the Qdrant client with configuration settings.
client = qdrant_client.QdrantClient(
    url=config['qdrant_url'],
    api_key=config['qdrant_api_key']
)

# Set up the vector store and index for document storage and retrieval.
vector_store = QdrantVectorStore(client=client, collection_name="ai-arxiv")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

with open("text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()

text_qa_template = PromptTemplate(text_qa_template_str)
# Initialize the language model with specified parameters.
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)

scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric()],
                        model_evaluator="gpt-3.5-turbo")

### NAIVE RAG ----------------------------------------------------------------------------------------------------------
query_engine_naive = index.as_query_engine(llm=llm,
                                           text_qa_template=text_qa_template,
                                           similarity_top_k=3)

get_llama_response_naive = make_get_llama_response(query_engine_naive)
# Score the responses
run_naive = scorer.score(benchmark, get_llama_response_naive)

print(run_naive.overall_scores)

remove_nul_chars_from_run_data(run_naive.run_data)
validate_api.upload_run(tonic_validate_project_key,
                        run = run_naive,
                        run_metadata = {"approach": "naive"})
### COHERE RERANK-------------------------------------------------------------------------------------------------------
cohere_api_key = config['cohere_api_key']
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)

query_engine_rerank = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[cohere_rerank],
)

get_llama_response_rerank = make_get_llama_response(query_engine_rerank)
run_rerank = scorer.score(benchmark, get_llama_response_rerank)

print(run_rerank.overall_scores)

remove_nul_chars_from_run_data(run_rerank.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_rerank,
                        run_metadata = {"approach": "cohere_rerank"})
### HyDE----------------------------------------------------------------------------------------------------------------
hyde = HyDEQueryTransform(include_original=True)
query_engine_hyde = TransformQueryEngine(query_engine_naive, hyde)

get_llama_response_hyde = make_get_llama_response(query_engine_hyde)

run_hyde = scorer.score(benchmark, get_llama_response_hyde)

print(run_hyde.overall_scores)

remove_nul_chars_from_run_data(run_hyde.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_hyde,
                        run_metadata = {"approach": "HyDE"})

### HyDE + RERANK ------------------------------------------------------------------------------------------------------
query_engine_hyde_rerank = TransformQueryEngine(query_engine_rerank, hyde)

get_llama_response_hyde_rerank = make_get_llama_response(query_engine_hyde_rerank)

run_hyde_rerank = scorer.score(benchmark, get_llama_response_hyde_rerank)

print(run_hyde_rerank.overall_scores)

remove_nul_chars_from_run_data(run_hyde_rerank.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_hyde_rerank,
                        run_metadata = {"approach": "HyDE + Cohere rerank"})

### MMR ----------------------------------------------------------------------------------------------------------------
query_engine_mmr = index.as_query_engine(vector_store_query_mode="mmr",
                                         similarity_top_k=3)

get_llama_response_mmr = make_get_llama_response(query_engine_mmr)

run_mmr = scorer.score(benchmark, get_llama_response_mmr)

print(run_mmr.overall_scores)

remove_nul_chars_from_run_data(run_mmr.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_mmr,
                        run_metadata = {"approach": "MMR"})