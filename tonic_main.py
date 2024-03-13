from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, PromptTemplate
import qdrant_client
from llama_index.llms.openai import OpenAI
import openai
from llama_index.embeddings.openai import OpenAIEmbedding
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
from llama_index.core import Settings
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

### SETUP --------------------------------------------------------------------------------------------------------------
# Load the config file
config = load_config('resources/config.json')

# Set the OpenAI API key for authentication.
openai.api_key = config['openai_api_key']
tonic_validate_api_key = config['tonic_validate_api_key']
tonic_validate_project_key = config['tonic_validate_project_key']
tonic_validate_benchmark_key = config['tonic_validate_benchmark_key']
validate_api = ValidateApi(tonic_validate_api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Traditional VDB
chroma_collection = chroma_client.get_collection("fidy_paper_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Sentence window VDB
chroma_collection_sentence_window = chroma_client.get_collection("fidy_paper_collection_sentence_window")
vector_store_sentence_window = ChromaVectorStore(chroma_collection=chroma_collection_sentence_window)
index_sentence_window = VectorStoreIndex.from_vector_store(vector_store=vector_store_sentence_window)
# Auto-merging VDB
chroma_collection_automerging = chroma_client.get_collection("fidy_paper_collection_automerging")
vector_store_automerging = ChromaVectorStore(chroma_collection=chroma_collection_automerging)
index_automerging = VectorStoreIndex.from_vector_store(vector_store=vector_store_automerging)

with open("resources/text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()

text_qa_template = PromptTemplate(text_qa_template_str)
# Initialize the language model with specified parameters.
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)

benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)

scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric()],
                        model_evaluator="gpt-3.5-turbo")

### Define query engines -------------------------------------------------------------------------------------------------
# Naive RAG
query_engine_naive = index.as_query_engine(llm = llm,
                                           text_qa_template=text_qa_template,
                                           similarity_top_k=3)

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

## LLM Rerank
from llama_index.core.postprocessor import LLMRerank
llm_rerank = LLMRerank(choice_batch_size=10, top_n=3)
query_engine_llm_rerank = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
)
# HyDE + LLM Rerank
query_engine_hyde_llm_rerank = TransformQueryEngine(query_engine_llm_rerank, hyde)

# Sentence window retrieval
query_engine_sentence_window = index_sentence_window.as_query_engine(text_qa_template=text_qa_template,
                                                                     similarity_top_k=3,
                                                                     llm = llm)

# Sentence window retrieval + Cohere rerank
query_engine_sentence_window_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[cohere_rerank],
)

# Sentence window retrieval + LLM Rerank
query_engine_sentence_window_llm_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
)

# Sentence window retrieval + HyDE
query_engine_sentence_window_hyde = TransformQueryEngine(query_engine_sentence_window, hyde)

# Sentence window retrieval + HyDE + Cohere Rerank
query_engine_sentence_window_hyde_rerank = TransformQueryEngine(query_engine_sentence_window_rerank, hyde)

# Sentence window retrieval + HyDE + LLM Rerank
query_engine_sentence_window_hyde_llm_rerank = TransformQueryEngine(query_engine_sentence_window_llm_rerank, hyde)

## Run experiments -------------------------------------------------------------------------------------------------------
# Dictionary of experiments, now referencing the predefined query engine objects
experiments = {
    # "NAIVE RAG": query_engine_naive,
    # "COHERE RERANK": query_engine_rerank,
    # "LLM RERANK": query_engine_llm_rerank,
    # "HyDE": query_engine_hyde,
    # "HyDE + Cohere Rerank": query_engine_hyde_rerank,
    # "HyDE + LLM Rerank": query_engine_hyde_llm_rerank
    # "Maximal Marginal Relevance (MMR)": query_engine_mmr,
    # "Multi Query": query_engine_multi_query,
    # "Multi Query + Cohere rerank + simple fusion": query_engine_multi_query_rerank,
    # "Sentence window retrieval": query_engine_sentence_window,
    # "Sentence window retrieval + Cohere rerank": query_engine_sentence_window_rerank,
    # "Sentence window retrieval + LLM Rerank": query_engine_sentence_window_llm_rerank,
    "Sentence window retrieval + HyDE": query_engine_sentence_window_hyde,
    "Sentence window retrieval + HyDE + Cohere Rerank": query_engine_sentence_window_hyde_rerank,
    "Sentence window retrieval + HyDE + LLM Rerank": query_engine_sentence_window_hyde_llm_rerank
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
                                            runs=1)  # Adjust the number of runs as needed

    # Append the results of this experiment to the master DataFrame
    all_experiments_results_df = pd.concat([all_experiments_results_df, experiment_results_df], ignore_index=True)

# Assuming all_experiments_results_df is your DataFrame
all_experiments_results_df['RetrievalPrecision'] = all_experiments_results_df['OverallScores'].apply(lambda x: x['retrieval_precision'])

# Check for normality and homogeneity of variances
# Skipping those checks here for brevity, but you should perform them in your analysis

from scipy.stats import shapiro

# Test for normality for each group
for exp in experiments.keys():
    stat, p = shapiro(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'])
    print(f"{exp} - Normality test: Statistics={stat}, p={p}")
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

from scipy.stats import levene

# Test for equal variances
stat, p = levene(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
print(f"Levene’s test: Statistics={stat}, p={p}")
if p > alpha:
    print('Equal variances across groups (fail to reject H0)')
else:
    print('Unequal variances across groups (reject H0)')

import scipy
# ANOVA
f_value, p_value = scipy.stats.f_oneway(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
print(f"ANOVA F-Value: {f_value}, P-Value: {p_value}")

# If ANOVA assumptions are not met, use Kruskal-Wallis
# h_stat, p_value_kw = stats.kruskal(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
# print(f"Kruskal-Wallis H-Stat: {h_stat}, P-Value: {p_value_kw}")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Extract just the relevant columns for easier handling
data = all_experiments_results_df[['Experiment', 'RetrievalPrecision']]

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(endog=data['RetrievalPrecision'], groups=data['Experiment'], alpha=0.05)
print(tukey_result)

# You can also plot the results to visually inspect the differences
import matplotlib.pyplot as plt
tukey_result.plot_simultaneous()
plt.show()

