import torch
from transformers import BertTokenizer
from multi_task_bert import MultiTaskBERT


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = MultiTaskBERT()
model.load_state_dict(torch.load("../models/multi_task_model.pth"))
model.eval()

def predict(sentence):

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        taskA_output, taskB_output = model(inputs["input_ids"], inputs["attention_mask"])
    
    # convert logits to class labels
    taskA_pred = torch.argmax(taskA_output, dim=-1).item()
    taskB_pred = torch.argmax(taskB_output, dim=-1).item()
    
    # interpret results
    spam_result = "Spam" if taskA_pred == 1 else "Not Spam"
    sentiment_result = "Positive" if taskB_pred == 1 else "Negative"

    return spam_result, sentiment_result

if __name__ == "__main__":
    # Test Inference
    test_sentence = "Win a free iPhone now!"
    spam_result, sentiment_result = predict(test_sentence)
    print(f"Sentence: {test_sentence}")
    print(f"Spam Classification: {spam_result}")
    print(f"Sentiment Analysis: {sentiment_result}")
