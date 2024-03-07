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
        response = query_engine.query(prompt)
        context = [x.text for x in response.source_nodes]
        return {
            "llm_answer": response.response,
            "llm_context_list": context
        }
    return get_llama_response