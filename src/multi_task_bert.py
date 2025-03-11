import torch
import torch.nn as nn
from transformers import BertModel

class MultiTaskBERT(nn.Module):
    def __init__(self):
        super(MultiTaskBERT, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Task A: Sentence Classification (spam)
        self.taskA_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        # Task B: Sentiment Analysis (positive/negative)
        self.taskB_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Task A (Spam Detection)
        taskA_output = self.taskA_classifier(pooled_output)
        # Task B (Sentiment Analysis)
        taskB_output = self.taskB_classifier(pooled_output)
        
        return taskA_output, taskB_output

if __name__ == "__main__":
    # Test to check if the model loads
    model = MultiTaskBERT()
    print(model)
