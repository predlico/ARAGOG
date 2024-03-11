from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, PromptTemplate
import qdrant_client
from llama_index.llms.openai import OpenAI
import openai
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import RetrievalPrecisionMetric, AnswerSimilarityMetric
from utils import make_get_llama_response, remove_nul_chars_from_run_data
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from utils import run_experiment, load_config
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import pandas as pd

### SETUP --------------------------------------------------------------------------------------------------------------
# Load the config file
config = load_config('resources/config.json')

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

with open("resources/text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()

text_qa_template = PromptTemplate(text_qa_template_str)
# Initialize the language model with specified parameters.
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)

benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)

scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric()],
                        model_evaluator="gpt-3.5-turbo")

### Define query engines -------------------------------------------------------------------------------------------------
# Naive RAG
query_engine_naive = index.as_query_engine(llm=llm, text_qa_template=text_qa_template, similarity_top_k=3)

# Cohere Rerank
cohere_rerank = CohereRerank(api_key=config['cohere_api_key'], top_n=3)  # Ensure top_n matches k in naive RAG for comparability
query_engine_rerank = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[cohere_rerank],
)

# HyDE
hyde = HyDEQueryTransform(include_original=True)
query_engine_hyde = TransformQueryEngine(query_engine_naive, hyde)

# HyDE + Cohere Rerank
query_engine_hyde_rerank = TransformQueryEngine(query_engine_rerank, hyde)

# Maximal Marginal Relevance (MMR)
query_engine_mmr = index.as_query_engine(vector_store_query_mode="mmr", similarity_top_k=3)

# Multi Query
vector_retriever = index.as_retriever(similarity_top_k=3)
retriever_multi_query = QueryFusionRetriever(
    [vector_retriever],
    similarity_top_k=3,
    llm=llm,
    num_queries=5,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True
)
query_engine_multi_query = RetrieverQueryEngine.from_args(retriever_multi_query, verbose=True)

# Multi Query + Cohere rerank + simple fusion
retriever_multi_query_rerank = QueryFusionRetriever(
    [vector_retriever],
    similarity_top_k=10,
    llm=llm,
    num_queries=5,
    mode="simple",
    use_async=False,
    verbose=True
)
query_engine_multi_query_rerank = RetrieverQueryEngine.from_args(retriever_multi_query_rerank, verbose=True, node_postprocessors=[cohere_rerank])

# Dictionary of experiments, now referencing the predefined query engine objects
experiments = {
    "NAIVE RAG": query_engine_naive,
    "COHERE RERANK": query_engine_rerank,
    "HyDE": query_engine_hyde,
    "HyDE + Cohere Rerank": query_engine_hyde_rerank,
    "Maximal Marginal Relevance (MMR)": query_engine_mmr,
    "Multi Query": query_engine_multi_query,
    "Multi Query + Cohere rerank + simple fusion": query_engine_multi_query_rerank,
    # Add more experiments as needed
}

# Initialize an empty DataFrame to collect results from all experiments
all_experiments_results_df = pd.DataFrame(columns=['Run', 'Experiment', 'OverallScores'])

# Loop through each experiment configuration, run it, and collect results
for experiment_name, query_engine in experiments.items():
    experiment_results_df = run_experiment(experiment_name,
                                            query_engine,
                                            scorer,
                                            benchmark,
                                            validate_api,
                                            config['tonic_validate_project_key'],
                                            upload_results=True,
                                            runs=5)  # Adjust the number of runs as needed

    # Append the results of this experiment to the master DataFrame
    all_experiments_results_df = pd.concat([all_experiments_results_df, experiment_results_df], ignore_index=True)

# Check for normality and homogeneity of variances
# Skipping those checks here for brevity, but you should perform them in your analysis

# Unfinished, needs some more love
#### Hybrid search -----------------------------------------------------------------------------------------------------
# QDRANT DOESNT WORK WITH BM25 - https://github.com/run-llama/llama_index/issues/9024
from llama_index.retrievers.bm25 import BM25Retriever
from datasets import load_dataset
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
# Load the AI research papers dataset from the Hugging Face datasets library
# dataset = load_dataset("jamescalam/ai-arxiv")
# # Convert the 'train' split of the dataset into a pandas DataFrame for easier manipulation
# df = pd.DataFrame(dataset['train'])
# documents = []
# # For each paper content in the filtered DataFrame, create a Document object and append it to the list
# for content in df['content']:
#     doc = Document(text=content)
#     documents.append(doc)
# parser = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
# index = VectorStoreIndex.from_documents(documents, transformations=[parser])
#
# vector_retriever = index.as_retriever(similarity_top_k=3)
# bm25_retriever = BM25Retriever.from_defaults(
#     docstore=index.docstore, similarity_top_k=2
# )

retriever_hybrid = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=1,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    verbose=True
)

query_engine_hybrid = RetrieverQueryEngine.from_args(retriever_hybrid,
                                                     verbose = True)

get_llama_response_hybrid = make_get_llama_response(query_engine_hybrid)
run_hybrid= scorer.score(benchmark, get_llama_response_hybrid)
print(run_hybrid.overall_scores)

remove_nul_chars_from_run_data(run_hybrid.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_hybrid,
                        run_metadata = {"approach": "Hybrid (vector + BM25)"})

#### Hybrid search + RERANK --------------------------------------------------------------------------------------------
retriever_hybrid = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=1,  # set this to 1 to disable query generation
    mode="simple",
    verbose=True
)

query_engine_hybrid_rerank = RetrieverQueryEngine.from_args(retriever_hybrid,
                                                     verbose = True,
                                                     node_postprocessors=[cohere_rerank])

get_llama_response_hybrid_rerank = make_get_llama_response(query_engine_hybrid_rerank)
run_hybrid_rerank= scorer.score(benchmark, get_llama_response_hybrid_rerank)
print(run_hybrid_rerank.overall_scores)

remove_nul_chars_from_run_data(run_hybrid_rerank.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_hybrid_rerank,
                        run_metadata = {"approach": "Hybrid (vector + BM25) + Cohere rerank"})

#### Hybrid search + RERANK + Multi query-------------------------------------------------------------------------------
retriever_hybrid = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=2,
    num_queries=5,  # set this to 1 to disable query generation
    mode="simple",
    verbose=True
)

query_engine_hybrid_rerank_mq = RetrieverQueryEngine.from_args(retriever_hybrid,
                                                     verbose = True,
                                                     node_postprocessors=[cohere_rerank])

get_llama_response_hybrid_rerank_mq = make_get_llama_response(query_engine_hybrid_rerank_mq)
run_hybrid_rerank_mq= scorer.score(benchmark, get_llama_response_hybrid_rerank_mq)
print(run_hybrid_rerank_mq.overall_scores)

remove_nul_chars_from_run_data(run_hybrid_rerank_mq.run_data)
validate_api.upload_run(tonic_validate_project_key, run = run_hybrid_rerank_mq,
                        run_metadata = {"approach": "Hybrid (vector + BM25) + Cohere rerank + Multi query"})

