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

openai.api_key  = os.environ['OPENAI_API_KEY']

# Load the dataset
dataset = load_dataset("jamescalam/ai-arxiv")

# Convert the 'train' split of the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Titles of the papers you want to include
selected_titles = [
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Measuring Massive Multitask Language Understanding",
    "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    "HellaSwag: Can a Machine Really Finish Your Sentence?"
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

##### QUESTIONS ##### - so far keeping it here becase node object is needed (how to get from Qdrant??)
keywords = ['llama']

# Adjusted filter to account for TextNode objects
included_nodes = [
    node for node in nodes
    if any(keyword in node.text.lower() for keyword in keywords)
]

# Convert the list to a DataFrame
df_nodes = pd.DataFrame(included_nodes)  # Adjust column name(s) as needed

# Export to Excel
df_nodes.to_excel('nodes.xlsx', index=False)


from llama_index.core.llama_dataset.generator import RagDatasetGenerator
dataset_generator = RagDatasetGenerator(
    nodes=included_nodes,
    llm=llm,
    num_questions_per_chunk=1,  # set the number of questions per nodes
)

rag_dataset = dataset_generator.generate_dataset_from_nodes()
print(rag_dataset.to_pandas().head())

# Assuming 'rag_dataset' is your dataset variable and it's already loaded with data
df = rag_dataset.to_pandas()
# Write the DataFrame to an Excel file
df.to_excel('rag_dataset.xlsx', index=False)

from llama_index.core.llama_pack import download_llama_pack

RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine, rag_dataset=rag_dataset, show_progress=True
)
benchmark_df = await rag_evaluator.arun(
    batch_size=20,  # batches the number of openai api calls to make
    sleep_time_in_seconds=1,  # seconds to sleep before making an api call
)
