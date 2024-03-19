import openai
import chromadb
import pandas as pd
from utils import run_experiment, load_config
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, PromptTemplate, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate.metrics import RetrievalPrecisionMetric, AnswerSimilarityMetric
from ExponentialBackoffFix.scorer_backoff_ValidateScorer import ValidateScorerBackoff
### SETUP --------------------------------------------------------------------------------------------------------------
# Load the config file
config = load_config('resources/config.json')

# Set the OpenAI API key for authentication.
openai.api_key = config['openai_api_key']
tonic_validate_api_key = config['tonic_validate_api_key']
tonic_validate_project_key = config['tonic_validate_project_key']
tonic_validate_benchmark_key = config['tonic_validate_benchmark_key']
validate_api = ValidateApi(tonic_validate_api_key)

# Service context
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
service_context = ServiceContext.from_defaults(llm = llm, embed_model = embed_model)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Traditional VDB
chroma_collection = chroma_client.get_collection("fidy_paper_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Sentence window VDB
chroma_collection_sentence_window = chroma_client.get_collection("fidy_paper_collection_sentence_window")
vector_store_sentence_window = ChromaVectorStore(chroma_collection=chroma_collection_sentence_window)
index_sentence_window = VectorStoreIndex.from_vector_store(vector_store=vector_store_sentence_window)
# Document summary VDB
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import get_response_synthesizer
# Initialize the sentence splitter
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
# Initialize the response synthesizer in 'tree_summarize' mode
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=False
)
# Cre
storage_context = StorageContext.from_defaults(persist_dir="Obelix")
doc_summary_index = load_index_from_storage(llm=llm,
                                              storage_context=storage_context
                                              )

# Prompt template
with open("resources/text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()

text_qa_template = PromptTemplate(text_qa_template_str)

# Tonic Validate setup
benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)
scorer = ValidateScorerBackoff(metrics=[RetrievalPrecisionMetric(), AnswerSimilarityMetric()],
                        model_evaluator="gpt-3.5-turbo")

### Define query engines -------------------------------------------------------------------------------------------------
# Naive RAG
query_engine_naive = index.as_query_engine(service_context = service_context,
                                           text_qa_template=text_qa_template,
                                           similarity_top_k=3)

# Cohere Rerank
cohere_rerank = CohereRerank(api_key=config['cohere_api_key'], top_n=3)  # Ensure top_n matches k in naive RAG for comparability
query_engine_rerank = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[cohere_rerank],
    service_context = service_context
)

# HyDE
hyde = HyDEQueryTransform(include_original=True)
query_engine_hyde = TransformQueryEngine(query_engine_naive, hyde)

# HyDE + Cohere Rerank
query_engine_hyde_rerank = TransformQueryEngine(query_engine_rerank, hyde)

# Maximal Marginal Relevance (MMR)
query_engine_mmr = index.as_query_engine(vector_store_query_mode="mmr",
                                         similarity_top_k=3,
                                         service_context = service_context)

# Multi Query
vector_retriever = index.as_retriever(similarity_top_k=3)
retriever_multi_query = QueryFusionRetriever(
    [vector_retriever],
    similarity_top_k=3,
    num_queries=5,
    llm=llm,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True
)
query_engine_multi_query = RetrieverQueryEngine.from_args(retriever_multi_query,
                                                          verbose=True,
                                                          service_context = service_context)

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
query_engine_multi_query_rerank = RetrieverQueryEngine.from_args(retriever_multi_query_rerank,
                                                                 verbose=True,
                                                                 node_postprocessors=[cohere_rerank],
                                                                 service_context = service_context)

## LLM Rerank
llm_rerank = LLMRerank(choice_batch_size=10, top_n=3)
query_engine_llm_rerank = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
    service_context = service_context
)
# HyDE + LLM Rerank
query_engine_hyde_llm_rerank = TransformQueryEngine(query_engine_llm_rerank, hyde)

# Sentence window retrieval
query_engine_sentence_window = index_sentence_window.as_query_engine(text_qa_template=text_qa_template,
                                                                     similarity_top_k=3,
                                                                     service_context = service_context)

# Sentence window retrieval + Cohere rerank
query_engine_sentence_window_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[cohere_rerank],
    service_context = service_context
)

# Sentence window retrieval + LLM Rerank
query_engine_sentence_window_llm_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
    service_context = service_context
)

# Sentence window retrieval + HyDE
query_engine_sentence_window_hyde = TransformQueryEngine(query_engine_sentence_window, hyde)

# Sentence window retrieval + HyDE + Cohere Rerank
query_engine_sentence_window_hyde_rerank = TransformQueryEngine(query_engine_sentence_window_rerank, hyde)

# Sentence window retrieval + HyDE + LLM Rerank
query_engine_sentence_window_hyde_llm_rerank = TransformQueryEngine(query_engine_sentence_window_llm_rerank, hyde)

# Document summary index
query_engine_doc_summary = doc_summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
    text_qa_template=text_qa_template,
    similarity_top_k=3,
    service_context = service_context,
    verbose=True
)

from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever
)
retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    similarity_top_k=5
)

# use retriever as part of a query engine
from llama_index.core.query_engine import RetrieverQueryEngine

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer

)

# query
response = query_engine.query("What is fine-tuning??")
print(response)
len(response.source_nodes)






query_engine_doc_summary = index_doc_summary.as_query_engine(text_qa_template=text_qa_template,
                                                             similarity_top_k=3,
                                                             service_context = service_context)

# HyDE + Document summary index
query_engine_hyde_doc_summary = TransformQueryEngine(query_engine_doc_summary, hyde)

# Document summary index + Cohere Rerank
query_engine_doc_summary_rerank = index_doc_summary.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[cohere_rerank],
    service_context=service_context
)

# Document summary index + LLM Rerank
query_engine_doc_summary_llm_rerank = index_doc_summary.as_query_engine(
    similarity_top_k=10,
    text_qa_template=text_qa_template,
    node_postprocessors=[llm_rerank],
    service_context = service_context
)

# Document summary index + HyDE + LLm Rerank
query_engine_hyde_doc_summary_llm_rerank = TransformQueryEngine(query_engine_doc_summary_llm_rerank, hyde)

# Document summary index + HyDE + Cohere Rerank
query_engine_hyde_doc_summary_rerank = TransformQueryEngine(query_engine_doc_summary_rerank, hyde)

## Run experiments -------------------------------------------------------------------------------------------------------
# Dictionary of experiments, now referencing the predefined query engine objects
experiments = {
    # "Classic VDB + Naive RAG": query_engine_naive,
    # "Classic VDB + Cohere Rerank": query_engine_rerank
    "Classic VDB + LLM Rerank": query_engine_llm_rerank,
    # "Classic VDB + HyDE": query_engine_hyde,
    # "Classic VDB + HyDE + Cohere Rerank": query_engine_hyde_rerank,
    # "Classic VDB + HyDE + LLM Rerank": query_engine_hyde_llm_rerank
    # "Classic VDB + Maximal Marginal Relevance (MMR)": query_engine_mmr,
    # "Classic VDB + Multi Query": query_engine_multi_query,
    # "Classic VDB + Multi Query + Cohere rerank": query_engine_multi_query_rerank,
    # "Sentence window retrieval": query_engine_sentence_window,
    # "Sentence window retrieval + Cohere rerank": query_engine_sentence_window_rerank,
    # "Sentence window retrieval + LLM Rerank": query_engine_sentence_window_llm_rerank,
    # "Sentence window retrieval + HyDE": query_engine_sentence_window_hyde,
    "Sentence window retrieval + HyDE + Cohere Rerank": query_engine_sentence_window_hyde_rerank,
    # "Sentence window retrieval + HyDE + LLM Rerank": query_engine_sentence_window_hyde_llm_rerank,
    "Document summary index": query_engine_doc_summary
    # "Document summary index + Cohere Rerank": query_engine_doc_summary_rerank,
    # "Document summary index + LLM Rerank": query_engine_doc_summary_llm_rerank
    # "Document summary index + HyDE": query_engine_hyde_doc_summary,
    # "Document summary index + HyDE + Cohere Rerank": query_engine_hyde_doc_summary_rerank
    # "Document summary index + HyDE + LLM Rerank": query_engine_hyde_doc_summary_llm_rerank
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
                                            runs=3)  # Adjust the number of runs as needed

    # Append the results of this experiment to the master DataFrame
    all_experiments_results_df = pd.concat([all_experiments_results_df, experiment_results_df], ignore_index=True)

# Assuming all_experiments_results_df is your DataFrame
all_experiments_results_df['RetrievalPrecision'] = all_experiments_results_df['OverallScores'].apply(lambda x: x['retrieval_precision'])

# Check for normality and homogeneity of variances
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
