import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import BertTokenizer
from multi_task_bert import MultiTaskBERT

# sample data
train_sentences = ["Buy now and get 50% off!", "I love this movie.", "This is a scam.", "Amazing product!"]
train_labels_taskA = [1, 0, 1, 0]  # spam (1) or not spam (0)
train_labels_taskB = [0, 1, 0, 1]  # negative (0) or positive (1)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer(train_sentences, padding=True, truncation=True, return_tensors="pt")
labels_taskA = torch.tensor(train_labels_taskA)
labels_taskB = torch.tensor(train_labels_taskB)

model = MultiTaskBERT()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# training loop
epochs = 3
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # forward pass
    taskA_output, taskB_output = model(inputs["input_ids"], inputs["attention_mask"])
    
    # compute loss for task A and B
    lossA = criterion(taskA_output, labels_taskA)
    lossB = criterion(taskB_output, labels_taskB)
    
    # combine losses
    total_loss = lossA + lossB
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {total_loss.item():.4f}")

# make sure the models directory exists
os.makedirs("../models", exist_ok=True)

torch.save(model.state_dict(), "../models/multi_task_model.pth")
print("Model saved successfully!")
