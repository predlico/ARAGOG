# ARAGOG - Advanced Retrieval Augmented Generation Output Grading :spider:

This repository contains the code, data, and analysis for our study [link later] on advanced Retrieval-Augmented Generation (RAG) techniques. It's part of our scientific paper investigating the efficacy of various RAG techniques in enhancing the precision and contextual relevance of LLMs.

## Repository Structure

- `eval_questions/`: Contains a JSON file with 107 QA pairs used in the evaluation.
- `papers_for_questions/`: Holds a collection of AI-ArXiv papers that were utilized for creating the 107 QA pairs.
- `resources/`: Includes essential resources like the prompt template and configuration files. Note: Actual config files need API keys and other settings to be filled out.
- `main.py`: The main script where experiments are defined and executed.
- `res_analysis.ipynb`: A Jupyter notebook for in-depth analysis of the final experimental results.
- `utils.py`: Helper functions supporting various operations within the repository.
- `vector_db.py`: Scripts for setting up different vector databases, such as Classic VDB, Sentence-window, and Document Summary.
- `final_results.xlsx`: Spreadsheet containing the final results from our experiments, shared for transparency and scientific verification.

## Getting Started

To replicate our experiments or to analyze our results, please ensure to fill in the necessary API keys and other configurations by renaming `config_template.json` to `config.json` and completing the required fields.

## Results examination

The `res_analysis.ipynb` notebook provides a detailed examination of the experimental results stored in `final_results.xlsx`. 

## Full replication

To set up vector databases for experiments, run the `vector_db.py` script. Subsequently, execute `main.py` to perform the experiments. Post-experimentation, use `res_analysis.ipynb` for analyzing the results. Helper functions in `utils.py` are employed across scripts to streamline processes.

## Contribution

Contributions are welcome. For any changes or enhancements, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the [MIT License](LICENSE).
