import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BaselineModel(nn.Module):
    def __init__(self, num_labels, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(BaselineModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Usamos AutoModelForSequenceClassification diretamente
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Certifique-se que todos os parâmetros estão com grad ativo para fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Access the embeddings module
        embeddings_module = self.model.bert.embeddings.position_embeddings

        # Iterate through named parameters of the embeddings module and freeze if not position_embeddings
        for p in embeddings_module.parameters():
            p.requires_grad = False
        
    def forward(self, input_ids, attention_mask=None):
        # O output já inclui logits diretamente
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        return logits
