from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_relevancy,
    answer_correctness,
    answer_similarity,
    answer_relevancy
)
import os
import json


def load_eval_data(dir_path, file_names):
    """
    Loads evaluation data from a set of JSON files containing questions and ground truths.

    Parameters:
    - dir_path (str): The directory path where the JSON files are stored.
    - file_names (list): A list of strings representing the names of the JSON files to load.

    Returns:
    - dict: A dictionary containing two keys: 'questions' and 'ground_truths'. Each key maps to a list
            where 'questions' contains all the questions loaded from the files, and 'ground_truths'
            contains all the corresponding ground truth answers.
    """
    all_data = {'questions': [], 'ground_truths': []}
    for file_name in file_names:
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r') as json_file:
            content = json.load(json_file)
            all_data['questions'].extend(content["questions"])
            all_data['ground_truths'].extend(content["ground_truths"])
    return all_data


def run_evaluations(all_data, llm, query_engine):
    """
    Runs evaluations on a set of questions using a specified language model and query engine, then returns the results as a pandas DataFrame.

    Parameters:
    - all_data (dict): A dictionary containing 'questions' and 'ground_truths' lists.
    - llm: The language model used for generating answers. This should be an instance of an OpenAI or similar model that supports the .query() method.
    - query_engine: The query engine that processes each question and generates an answer. This engine uses the provided language model.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original questions, generated answers, contexts from where the answers were derived, and the ground truth answers.
    """
    answers = []
    contexts = []
    for query in all_data['questions']:
        query_result = query_engine.query(query)
        answers.append(query_result.response)
        contexts.append([node.node.text for node in query_result.source_nodes])
    rag_data = {
        "question": all_data['questions'],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": all_data["ground_truths"]
    }
    dataset = Dataset.from_dict(rag_data)
    result = evaluate(
        dataset=dataset,
        metrics=[answer_correctness, context_relevancy, faithfulness, answer_similarity],
    )
    return result.to_pandas()
