import json
def remove_nul_chars_from_string(s):
    """Remove NUL characters from a single string."""
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
        response = query_engine.query(prompt)
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

def run_experiment(experiment_name,
                   query_engine,
                   scorer,
                   benchmark,
                   validate_api,
                   project_key,
                   upload_results=True):
    get_llama_response_func = make_get_llama_response(query_engine)
    run = scorer.score(benchmark, get_llama_response_func)
    print(f"{experiment_name} Overall Scores:", run.overall_scores)
    remove_nul_chars_from_run_data(run.run_data)

    if upload_results:
        validate_api.upload_run(project_key, run=run, run_metadata={"approach": experiment_name})
    else:
        print(f"Skipping upload for {experiment_name}.")
