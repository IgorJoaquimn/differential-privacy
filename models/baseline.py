from transformers import DistilBertModel, DistilBertTokenizer
import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self, model_name="distilbert/distilbert-base-uncased"):
        super(BaselineModel, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.original_emb = self.model.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
