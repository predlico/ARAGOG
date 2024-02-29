from datasets import load_dataset
import pandas as pd
from llama_index.core import Document

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

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.chunk_size = 512
Settings.chunk_overlap = 50

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents,
                                        service_context=service_context)
query_engine = index.as_query_engine()
response = query_engine.query(
    "Decribe pre-training process of BERT model"
)
print(str(response))

# index.storage_context.persist(persist_dir="vector_store")

eval_questions = []
with open('test_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)

print(eval_questions)

from trulens_eval import Tru, TruLlama
tru = Tru()

tru.reset_database()

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

with tru_recorder as recording:
    for question in eval_questions[:10]:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()
tru.run_dashboard()





# UTILS - to    be moved to a separate file
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)

import numpy as np
from trulens_eval.feedback import Groundedness
openai = OpenAI()
qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

#grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder