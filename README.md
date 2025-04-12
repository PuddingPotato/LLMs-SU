## Development of an AI Model for Course Recommendation and Academic Planning in University Curriculums Using Large Language Models
A Retrieval-Augmented Generation (RAG)-based system for intelligent course recommendation and study planning using Large Language Models (LLMs), powered by LangChain.

##  Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to support AI-driven course recommendation and study planning. The system connects multiple components—including data retrieval, embeddings, and large language models—using the LangChain framework, with LangSmith for model monitoring and evaluation.

##  Models Used
Large Language Models (LLMs):

- LLaMA 3.3 70B

- TYPHOON AI 2 70B

Embedding Models:

- Cohere Multilingual 3.0

- BGE M3

##  Key Findings
Cohere Multilingual 3.0 performs better than BGE M3 in retrieving relevant academic data.

The best-performing combination:

- Embedding: Cohere Multilingual 3.0

- LLM: TYPHOON AI 2 70B

Achieved precision = 0.825, demonstrating high-quality information filtering.

## Data Source
- Course information was collected via web scraping using BeautifulSoup and Selenium.

- The source website: Faculty of Science, Silpakorn University.

- The dataset covers the latest academic year: 2567 (2024).

## Tech Stack
- LangChain: Orchestrates retrieval, generation, and overall pipeline.

- LangSmith: Used for monitoring, debugging, and tracking model performance.

- BeautifulSoup + Selenium: For scraping structured course data from university webpages.

