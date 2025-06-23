import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings using attention mask."""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BaselineModel(nn.Module):
    def __init__(self, num_labels, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(BaselineModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        # Compute token embeddings
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, attention_mask)
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        # Pass through classifier for labels
        logits = self.classifier(sentence_embeddings)
        
        return logits
