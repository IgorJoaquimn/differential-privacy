from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-10, device=None):
    """Sample Gumbel noise with given shape."""
    U = torch.empty(shape, device=device).uniform_(eps, 1.0 - eps)
    return -torch.log(-torch.log(U))


def get_metric_truncated_exponential_mechanism(
    vectors: torch.Tensor,
    original_embedding: torch.Tensor,
    embed_norm: torch.Tensor,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Apply metric-based truncated exponential mechanism with Gumbel noise.

    Args:
        vectors (torch.Tensor): Input tensor of shape (B, T, D).
        original_embedding (torch.Tensor): Embedding matrix of shape (V, D).
        epsilon (float): Privacy parameter (smaller epsilon adds more noise).

    Returns:
        torch.Tensor: Selected indices of shape (B, T).
    """
    B, T, D = vectors.shape
    V = original_embedding.shape[0]

    # Normalize vectors for cosine similarity
    vectors_norm = F.normalize(vectors.view(B * T, D), p=2, dim=1)     # (B*T, D)

    # Compute cosine similarity between each vector and each embedding
    similarity = torch.mm(vectors_norm, embed_norm.T)               # (B*T, V)

    # Sample Gumbel noise and add scaled noise
    gumbel_noise = sample_gumbel(similarity.shape, device=similarity.device)
    noisy_scores = similarity + (2 / epsilon) * gumbel_noise            # (B*T, V)

    # Select index with maximum noisy score for each vector
    selected_indices = torch.argmax(noisy_scores, dim=1)                # (B*T,)

    # Gather the selected embeddings by index
    selected_vectors = original_embedding[selected_indices]              # (B*T, D)

    # Reshape back to (B, T, D)
    return selected_vectors.view(B, T, D)


class NoisyEmbedding(nn.Module):
    def __init__(self, original_embedding, epsilon=0.1):
        super().__init__()
        self.embedding = original_embedding
        self.normalized_embedding = F.normalize(original_embedding.weight, p=2, dim=1)           # (V, D)
        self.epsilon = epsilon

    def forward(self, input_ids):
        return get_metric_truncated_exponential_mechanism(
            self.embedding(input_ids),
            self.embedding.weight,
            self.normalized_embedding.to(input_ids.device),
            self.epsilon,
        )


class TEMModel(nn.Module):
    def __init__(self, num_labels=2, model_name="sentence-transformers/all-MiniLM-L6-v2", epsilon=5):
        super(TEMModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Usamos AutoModelForSequenceClassification diretamente
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.epsilon = epsilon

        # Replace the word embeddings with noisy embeddings
        self.original_emb = self.model.get_input_embeddings()
        self.model.set_input_embeddings(NoisyEmbedding(self.original_emb, epsilon))


        # Certifique-se que todos os parâmetros estão com grad ativo para fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True

        # Access the embeddings module
        embeddings_module = self.model.bert.embeddings.position_embeddings

        # Iterate through named parameters of the embeddings module and freeze if not position_embeddings
        for p in embeddings_module.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.logits
        return outputs
    
    def get_embeddings(self, input_ids):
        """
        Recebe input_ids e retorna o embedding tensor correspondente (com ruído aplicado).
        """
        # Passa os ids pelo embedding (que já está modificado com ruído)
        embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings
    
