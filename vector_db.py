from datasets import load_dataset
import pandas as pd
from llama_index.core import Document
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter

import os
import openai

openai.api_key = os.environ['OPENAI_API_KEY']


# Load the dataset
dataset = load_dataset("jamescalam/ai-arxiv")

# Convert the 'train' split of the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Titles of the papers you want to include
selected_titles = [
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    # "Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity",
    # "LLaMA: Open and Efficient Foundation Language Models",
    # "Measuring Massive Multitask Language Understanding",
    # "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    # "HellaSwag: Can a Machine Really Finish Your Sentence?"
]

# Filter the DataFrame to only include the specified titles
selected_papers = df[df['title'].isin(selected_titles)]

# Continue with your existing code but use 'selected_papers' instead of 'random_papers'
documents = []
for content in selected_papers['content']:
    doc = Document(text=content)
    documents.append(doc)


# parse nodes
parser = TokenTextSplitter(chunk_size = 512,
                           chunk_overlap = 50)

nodes = parser.get_nodes_from_documents(documents)

embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# llm = OpenAI(model="gpt-4-1106-preview", temperature=0.1)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)
client = qdrant_client.QdrantClient(
    # otherwise set Qdrant instance address with:
    url="https://e0ef990f-d23a-412d-8104-66300708e2ff.us-east4-0.gcp.cloud.qdrant.io:6333",
    # set API KEY for Qdrant Cloud
    api_key="ZaQf7Wh2IS3Vp9Tdn2HwCXw3Oyq3rxBMZkdf8rwVhuAKBwEYkjZ40A"
)

# client.delete_collection(collection_name="six_papers_arxiv_ai")
vector_store = QdrantVectorStore(client=client, collection_name="six_papers_arxiv_ai")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes,
                         storage_context=storage_context)

query_engine = index.as_query_engine(llm = llm)
# response = query_engine.query(
#     "Decribe pre-training process of BERT model"
# )
# print(str(response))





from datasets import Dataset
import json
import os

# Assuming all JSON files are in the same directory and have been downloaded to the local environment
file_names = [
    'questions_ground_truth_bert.json',
    'questions_ground_truth_distilbert.json',
    'questions_ground_truth_hellaswag.json',
    'questions_ground_truth_llama.json',
    'questions_ground_truth_mmlu.json',
    'questions_ground_truth_red_teaming.json'
]

# The directory where your JSON files are stored (this needs to be the path to your local directory)
dir_path = 'eval_questions/'

# Dictionary to store all data from the JSON files
all_data = {'questions': [], 'ground_truths': []}

# Loop over the file names and load the content
for file_name in file_names:
    # Construct the full file path
    file_path = os.path.join(dir_path, file_name)

    # Load the JSON file and add its content to the all_data dictionary
    with open(file_path, 'r') as json_file:
        content = json.load(json_file)
        all_data['questions'].extend(content["questions"])
        all_data['ground_truths'].extend(content["ground_truths"])

# Now you can access all the questions and ground truths with:
questions = all_data["questions"]
ground_truths = all_data["ground_truths"]
answers = []
contexts = []

# Inference
for query in questions:
    query_result = query_engine.query(query)
    answers.append(query_result.response)  # Assuming query_result itself contains the answer you need
    # Extracting and appending text content from source_nodes for each query
    contexts.append([node.node.text for node in query_result.source_nodes])

# To dict
rag_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(rag_data)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_relevancy,
    answer_correctness
)
# discuss which metrics to use - https://docs.ragas.io/en/v0.0.17/concepts/metrics/index.html
result = evaluate(
    dataset = dataset,
    metrics=[
        answer_correctness,
        context_relevancy,
        faithfulness
    ],
)

df = result.to_pandas()
df.to_excel("eval_results.xlsx")