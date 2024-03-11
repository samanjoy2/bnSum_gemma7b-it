# Bangla News Summarization with Gemma-7b (Instruct)

This GitHub repo outlines an approach for building a **Bangla News Summarization** project utilizing **Gemini-7b (Instrct)**.

### Dataset

**Link:** https://www.kaggle.com/datasets/prithwirajsust/bengali-news-summarization-dataset

## Reason for using this dataset:
1. Focuses on Bengali Abstractive News Summarization (BANS): This dataset is specifically designed for training models that can generate summaries of Bengali news articles, unlike datasets that focus on extractive summarization (copying sentences from the article).
2. Large corpus for Bengali NLP tasks: With over 19,000 articles and summaries, this dataset provides a valuable resource for training and evaluating Bengali Natural Language Processing (NLP) models, especially those focused on summarization.
3. Publicly available and well-documented: The dataset is hosted on Kaggle, a popular platform for data science, making it easily accessible for researchers and practitioners. The dataset description includes details about the data collection process and statistics about the articles and summaries.

## Model Choice
* **Model Name:** Gemma 7b (Instruct)
* **Model Link:** https://huggingface.co/google/gemma-7b-it

### Text-to-Text, Decoder-only
This architecture is commonly used for summarization tasks, where the model takes the article as input and generates a summary as output.
### Large Language Model (LLM)
LLMs are known for their ability to handle complex language tasks like summarization.
### Zero Shot Bangla Text Handling
Without any kind of finetuning, it was noticed that Gemma-7b understood Bengali quite well compared to other Models.

## Finetuning
# Prompt for Finetuning
```shell
prompt = f"""
<start_of_turn>
Provide a concise Bengali summary of the following news article, focusing on the most important information. 

Note:
Use only Bengali for the summary.
Stay objective and factual in your summary.

####

Article: {train_df["Text"].values[0]}

####
<end_of_turn>
"""
prompt
```

# Hi
