import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class BaselineModel(nn.Module):
    def __init__(self, num_labels, model_name="distilbert/distilbert-base-uncased"):
        super(BaselineModel, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.original_emb = self.model.distilbert.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
