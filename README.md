# Bangla News Summarization with Gemma-7b (Instruct)

This github repo outlines an approach for building a Bangla News Summarization project utilizing Gemini-7b (Instrct).

### Dataset

Link : https://www.kaggle.com/datasets/prithwirajsust/bengali-news-summarization-dataset

## Reason for using this dataset:
1. Focuses on Bengali Abstractive News Summarization (BANS): This dataset is specifically designed for training models that can generate summaries of Bengali news articles, unlike datasets that focus on extractive summarization (copying sentences from the article).
2. Large corpus for Bengali NLP tasks: With over 19,000 articles and summaries, this dataset provides a valuable resource for training and evaluating Bengali Natural Language Processing (NLP) models, especially those focused on summarization.
3. Publicly available and well-documented: The dataset is hosted on Kaggle, a popular platform for data science, making it easily accessible for researchers and practitioners. The dataset description includes details about the data collection process and statistics about the articles and summaries.

## Model Choice
Model: Gemma 7b (Instruct) https://huggingface.co/google/gemma-7b-it

1. Text-to-Text, Decoder-only: This architecture is commonly used for summarization tasks, where the model takes the article as input and generates a summary as output.
2. Large Language Model (LLM): LLMs are known for their ability to handle complex language tasks like summarization.
3. State-of-the-Art: Gemma-7b being state-of-the-art suggests it might outperform older models on this task.
