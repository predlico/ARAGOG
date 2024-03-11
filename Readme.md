# ARAGOG - Advanced Retrieval Augmented Generation Output Grading :spider:

This repository contains code and resources for our RAG (Retrieval-Augmented Generation) experiments. We are comparing different RAG approaches on the same dataset.

## General Guidelines

- Keep the code modular. This allows us to experiment more easily and potentially re-use the code for our client projects.

## Pre-Coding Tasks

### Dataset Selection

We have selected a dataset of [AI papers from arXiv](https://huggingface.co/datasets/jamescalam/ai-arxiv) for our experiments.

### Evaluation Framework Selection

We have chosen [Trulens](https://www.trulens.org/) as our evaluation framework.

## Stack

We will be utilizing LLaMAIndex wherever possible as our primary RAG stack. For approaches not encompassed by LLaMAIndex, we will be developing our own custom code.

## Coding Tasks

The tasks include setting up multiple vector databases and creating questions. The same set of questions will be evaluated across different types of RAG.

## Techniques to Run

We will be running the following techniques:

- Sentence-window retrieval
- Auto-merging retrieval
- Query expansion - hallucinate output
- Query expansion - create similar questions
- Reranker
- Fine-tune reranker
- Step back prompting
- Maximum marginal relevance
- Embedding adapters
- Hybrid search (multiple vectordbs, one sentences, one n-grams, one paragraphs..)

## Order of Operations

1. Step back prompting
2. Reranker
3. Query expansion - hallucinate output
4. Query expansion - create similar questions
5. Maximum marginal relevance

## Time Intensive Techniques

- Embedding adapters
- Fine-tune reranker

## Techniques that Might Require Re-indexing

- Sentence-window retrieval
- Auto-merging retrieval
