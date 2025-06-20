from transformers import DistilBertModel, DistilBertTokenizer
import torch
import torch.nn as nn


def add_non_negative_exponential_noise(vector, original_embedding, epsilon=0.1):
    """
    Adds noise to a vector following a specific procedure:
    

    Args:
        vector (torch.Tensor): The input vector (or tensor of any shape).
        epsilon (float): The privacy parameter. Smaller epsilon typically means more noise.

    Returns:
        torch.Tensor: The vector with added noise.
    """
 
    return vector + torch.abs(torch.randn_like(vector)) * epsilon

class NoisyEmbedding(nn.Module):
    def __init__(self, original_embedding, epsilon=0.1):
        super().__init__()
        self.embedding = original_embedding
        self.epsilon = epsilon

    def forward(self, input_ids):
        return add_non_negative_exponential_noise(self.embedding(input_ids), self.embedding.weight, self.epsilon)

class TEMModel(nn.Module):
    def __init__(self, model_name="distilbert/distilbert-base-uncased", epsilon=0.1):
        super(TEMModel, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.epsilon = epsilon

        # Replace the word embeddings with noisy embeddings
        self.original_emb = self.model.embeddings.word_embeddings
        self.model.embeddings.word_embeddings = NoisyEmbedding(self.original_emb, epsilon)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
