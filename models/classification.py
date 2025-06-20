import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, embedding_model, embed_dim, num_classes):
        super().__init__()
        self.embedding = embedding_model
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)              # [B, T, D]
        pooled = embedded.mean(dim=1)             # [B, D]
        return self.fc(pooled)