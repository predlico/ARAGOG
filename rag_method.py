from .vector_db import VectorDB

class RAGMethod:
    def __init__(self, name, vector_db: VectorDB):
        self.name = name
        self.vector_db = vector_db
        # Initialize your RAG method here

    def generate(self, query):
        # Retrieve relevant documents from the vector database
        retrieval = self.vector_db.retrieve(query)
        # Implement your generation method here
        pass