import json
import pandas as pd
def remove_nul_chars_from_string(s):
    """Remove NUL characters from a single string."""
    """ Nece """
    return s.replace('\x00', '')

def remove_nul_chars_from_run_data(run_data):
    """Iterate over all fields of RunData to remove NUL characters."""
    for run in run_data:
        run.reference_question = remove_nul_chars_from_string(run.reference_question)
        run.reference_answer = remove_nul_chars_from_string(run.reference_answer)
        run.llm_answer = remove_nul_chars_from_string(run.llm_answer)
        run.llm_context = [remove_nul_chars_from_string(context) for context in run.llm_context]

def make_get_llama_response(query_engine):
    def get_llama_response(prompt):
        print(prompt)
        time.sleep(5)
        response = query_engine_doc_summary.query(prompt)
        context = [x.text for x in response.source_nodes]
        return {
            "llm_answer": response.response,
            "llm_context_list": context
        }
    return get_llama_response

def chunked_iterable(iterable, size):
    """Yield successive size chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# Function to load and validate configuration settings
def load_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    # Example validation, extend as needed
    required_keys = ['openai_api_key', 'qdrant_url', 'qdrant_api_key']
    if not all(key in config for key in required_keys):
        raise ValueError("Config file is missing required keys.")
    return config

def run_experiment(experiment_name, query_engine, scorer, benchmark, validate_api, project_key, upload_results=True, runs=5):
    # List to store results dictionaries
    results_list = []

    for i in range(runs):
        get_llama_response_func = make_get_llama_response(query_engine)
        run = scorer.score(benchmark,
                           get_llama_response_func,
                           callback_parallelism=3,
                           scoring_parallelism=3)
        print(f"{experiment_name} Run {i+1} Overall Scores:", run.overall_scores)
        remove_nul_chars_from_run_data(run.run_data)

        # Add this run's results to the list
        results_list.append({'Run': i+1, 'Experiment': experiment_name, 'OverallScores': run.overall_scores})

        if upload_results:
            validate_api.upload_run(project_key, run=run, run_metadata={"approach": experiment_name, "run_number": i+1})
        else:
            print(f"Skipping upload for {experiment_name} Run {i+1}.")

    # Create a DataFrame from the list of results dictionaries
    results_df = pd.DataFrame(results_list)

    # Return the DataFrame containing all the results
    return results_df

