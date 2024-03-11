# Bangla News Summarization with Gemma-7b (Instruct)

This GitHub repo outlines an approach for building a **Bangla News Summarization** project utilizing **Gemini-7b (Instrct)**.
* The finetuned model is available on: https://huggingface.co/samanjoy2/gemma7b-it_banglaNewsSum

### Dataset

* **Link:** https://www.kaggle.com/datasets/prithwirajsust/bengali-news-summarization-dataset
* We utilized the same data splits used by https://link.springer.com/chapter/10.1007/978-981-33-4673-4_4 for direct comparison.

## Reason for using this Dataset
1. **Focuses on Bengali Abstractive News Summarization (BANS):** This dataset is specifically designed for training models that can generate summaries of Bengali news articles, unlike datasets that focus on extractive summarization (copying sentences from the article).
2. **Large corpus for Bengali NLP tasks:** With over 19,000 articles and summaries, this dataset provides a valuable resource for training and evaluating Bengali Natural Language Processing (NLP) models, especially those focused on summarization.
3. **Publicly available and well-documented:** The dataset is hosted on Kaggle, a popular platform for data science, making it easily accessible for researchers and practitioners. The dataset description includes details about the data collection process and statistics about the articles and summaries.

## Model Choice
* **Model Name:** Gemma 7b (Instruct)
* **Model Link:** https://huggingface.co/google/gemma-7b-it

### Text-to-Text, Decoder-only
This architecture is commonly used for summarization tasks, where the model takes the article as input and generates a summary as output.
### Large Language Model (LLM)
LLMs are known for their ability to handle complex language tasks like summarization.
### Zero Shot Bangla Text Handling
Without any kind of finetuning, it was noticed that Gemma-7b understood Bengali quite well compared to other Models.

## How to run
1. Clone this repo
   ```python
   git clone https://github.com/samanjoy2/bnSum_gemma7b-it.git
   ```
2. Replace the *kaggle.json* file with yours one. Link: https://www.kaggle.com/settings -> API -> Create New Token
3. [Training] Run the *training.ipynb*. It was trained with *NVIDIA RTX A6000* from [vast.ai](https://vast.ai/).
4. [Inference] Run the *inference.ipynb*. Can be done with ~8GB VRAM. Was used *NVIDIA GeForce RTX 3090*.

## **Finetuning**
## **Loading** Gemma 7b Model with **Quantization** Configuration (Bits and Bytes)
**Quantization** configuration is applied during the **loading** process. **Quantization** is a technique used to reduce the memory footprint and computational cost of **neural networks** by representing weights and activations with lower precision data types, such as **8-bit integers** or **bfloat16**. **Bits and Bytes** is a library for model **quantization** and **compression** developed by Hugging Face. It offers tools and techniques for reducing the size of **neural network** models, making them more memory and computationally efficient.
In the provided code, **Bits and Bytes** is used for **quantization** of the Gemma 7b model. The **configuration** used in the code specifies parameters such as whether to **load** weights in **4-bit format**, the type of **quantization method** to use, and the data type for computation during **quantization** (**bfloat16**).
By applying **quantization** techniques from the **Bits and Bytes** library, the **loaded** model can be trained and **fine-tuned** more efficiently while still maintaining performance levels.

## **Defining** LoraConfig for **LoRA** Layer Integration

**LoraConfig** is **defined** to integrate **LoRA** (Learned Optimizer for Rethinking Attention) layers into the model.
**LoRA** is a technique that aims to improve the **attention** mechanism in **transformer** models by learning the optimal **attention** pattern for each layer dynamically during training. This can lead to more efficient and effective **attention** mechanisms.

## **Preparing** for Training with **Peft** and **LoRA**:
The model is **prepared** for training with **Peft** (Parameter Efficient **Fine-Tuning**) and **LoRA** techniques.
**Peft** is a method for **fine-tuning** pre-trained language models with reduced computational and memory requirements compared to traditional **fine-tuning** approaches. It aims to achieve similar performance while using fewer resources.
By integrating **Peft** and **LoRA** into the model **configuration**, the model is set up to undergo efficient **fine-tuning** with improved **attention** mechanisms.


### Prompt for Finetuning
```python
<start_of_turn>
Provide a concise Bengali summary of the following news article, focusing on the most important information. 

Note:
Use only Bengali for the summary.
Stay objective and factual in your summary.

####

Article: {data_point["Text"]}

####
<end_of_turn>

<start_of_turn>
####

Summary: {data_point["Summary"]} 

####
<end_of_turn>
```

### Hyperparameters
``` python
save_strategy="steps"
evaluation_strategy="steps"
per_device_train_batch_size=16
per_device_eval_batch_size=32
gradient_accumulation_steps=16
num_train_epochs=5
logging_steps=20
eval_steps=20
save_steps=20
warmup_steps=100
learning_rate=2e-4
fp16=True
optim="paged_adamw_8bit"
lr_scheduler_type="cosine"
warmup_ratio=0.01
report_to="none"
save_total_limit=3
load_best_model_at_end=True
```

### Training Output

``` python
TrainOutput(global_step=260, training_loss=3.2770693999070386, metrics={'train_runtime': 16852.8414, 'train_samples_per_second': 3.966, 'train_steps_per_second': 0.015, 'total_flos': 7.048804415292273e+17, 'train_loss': 3.2770693999070386, 'epoch': 4.98})
```
## Results
```python
Predicted Summary: ময়মনসিংহে বাস ও অটোরিকশার সংঘর্ষে পাঁচজন নিহত
True Label: ময়মনসিংহে বাসঅটোরিকশা সংঘর্ষে নিহত ৫

Predicted Summary: নারায়ণগঞ্জে সাত খুনের ঘটনায় পুলিশের গ্রেপ্তার
True Label: নাগঞ্জে ৭ খুন: আরেকজন গ্রেপ্তার

Predicted Summary: নিজস্ব অর্থে বহু প্রতীক্ষিত পদ্মা সেতুর কাজ চলছে পুরোদমে।
True Label: নিজস্ব অর্থে এগিয়ে যাচ্ছে পদ্মা সেতুর কাজ
```

### Analysis
* Many times the LLM didn't follow the prompt template, as a result, there was sometimes no output.
* Coulnt generate new words that were not in the main article which was seen in the dataset.
* Our model's BLEU Score is 0.4 which is higher than https://link.springer.com/chapter/10.1007/978-981-33-4673-4_4 [BLEU: 0.3]. [Note: We skipped the sentences that the model could not generate any result]
