# Sentence Transformer with Multi-Task Learning

## Description
Sentence Transformer that generates embeddings and and is fine-tuned using Multi-Task Learning (MTL) to handle:
- Task A: Sentence Classification (spam/not spam)
- Task B: Sentiment Analysis (postive/negative/neutral)

## Technologies
- Hugging Face Transformers for pre-trained BERT
- PyTorch for model training

## Run with Docker
To run example inference in a Docker container, run the following in the root directory:
- 1 - `docker build -t multi-task-bert .`
- 2 - `docker run --rm multi-task-bert`