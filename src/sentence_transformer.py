import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_sentence_embedding(sentence):
    """
    Converts a sentence into a fixed-length vector using BERT.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    return embedding

# Test the function with an example
if __name__ == "__main__":
    test_sentence = "Transformers are revolutionizing NLP."
    embedding = get_sentence_embedding(test_sentence)
    print(f"Sentence Embedding (first 3 values): {embedding[:3]}")
