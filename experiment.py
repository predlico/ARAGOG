from .rag_method import RAGMethod

class Experiment:
    def __init__(self, rag_methods: list[RAGMethod], questions: list[str]):
        self.rag_methods = rag_methods
        self.questions = questions

    def run(self):
        results = {}
        for question in self.questions:
            for rag_method in self.rag_methods:
                generation = rag_method.generate(question)
                results[(question, rag_method.vector_db.name, rag_method.name)] = generation
        return results