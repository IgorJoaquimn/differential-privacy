from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import torch.nn as nn


def add_non_negative_exponential_noise(vector, epsilon=0.1):
    """
    Adds noise to a vector following a specific procedure:
    1. Samples a vector 'v' from a multivariate normal distribution.
    2. Normalizes 'v' to constrain it within the unit ball (unit vector).
    3. Samples a magnitude 'l' from a Gamma distribution.
    4. The noisy vector is then 'l * v'.

    This method is often used for L2-based differential privacy or to
    generate noise with a specific norm distribution.

    Args:
        vector (torch.Tensor): The input vector (or tensor of any shape).
        epsilon (float): The privacy parameter. Smaller epsilon typically means more noise.

    Returns:
        torch.Tensor: The vector with added noise.
    """
 
    # Get the embedding dimension (always the last dimension)
    embed_dim = vector.shape[-1]
       
    # 1. Sample a vector-valued random variable v from the multivariate normal distribution:
    # We sample each component independently from a standard normal distribution (mean=0, std=1).
    # The shape should match the input vector's full shape.
    v = torch.randn(vector.shape, device=vector.device) # randn generates N(0,1)

    # 2. The vector v is then normalized to constrain it in the unit ball.
    # This means dividing by its L2 norm to make it a unit vector.
    # The norm needs to be calculated across the 'embed_dim' (last dimension).
    # Add a small epsilon to the denominator to prevent division by zero if v is all zeros.
    norm_v = torch.linalg.norm(v, ord=2, dim=-1, keepdim=True) # dim=-1 ensures norm is computed per embedding
    v_normalized = v / (norm_v + 1e-8) # L2 normalization

    # 3. Next, we sample a magnitude l from the Gamma distribution where θ = 1/ε
    #    and n is the embedding dimensionality.
    # In PyTorch, Gamma distribution is parameterized by 'concentration' (alpha) and 'rate' (beta).
    # Concentration (alpha or k) is the embedding dimensionality (n)
    alpha = float(embed_dim)
    # Rate (beta or 1/theta) is epsilon
    beta = epsilon # PyTorch's Gamma(concentration, rate)

    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
    
    # Sample a magnitude 'l'. We need one magnitude per independent embedding.
    # The shape of `magnitude_l` should be `(..., 1)` to broadcast correctly
    # with `v_normalized` which has shape `(..., embed_dim)`.
    
    # The number of independent embeddings to sample magnitudes for is the product
    # of all dimensions *except* the last one (embed_dim).
    magnitude_l_shape = vector.shape[:-1] # All dimensions except the last one
    
    # Sample magnitudes for each individual embedding in the batch/sequence
    magnitude_l = gamma_dist.sample(magnitude_l_shape).to(vector.device)
    
    # Reshape to (..., 1) to enable broadcasting over the embedding dimension
    magnitude_l = magnitude_l.unsqueeze(-1)

    # 4. A sample noisy vector at the privacy parameter ε is therefore output as lv
    noise = magnitude_l * v_normalized

    # Add the noise to the vector
    perturbed_vector = vector + noise
    return perturbed_vector

class NoisyEmbedding(nn.Module):
    def __init__(self, original_embedding, epsilon=0.1):
        super().__init__()
        self.embedding = original_embedding
        self.epsilon = epsilon

    def forward(self, input_ids):
        return add_non_negative_exponential_noise(self.embedding(input_ids), self.epsilon)

class MadlibModel(nn.Module):
    def __init__(self, num_labels, model_name="distilbert/distilbert-base-uncased", epsilon=0.1):
        super(MadlibModel, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.epsilon = epsilon

        # Replace the word embeddings with noisy embeddings
        self.original_emb = self.model.distilbert.embeddings.word_embeddings
        self.model.distilbert.embeddings.word_embeddings = NoisyEmbedding(self.original_emb, epsilon)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Pass along any extra kwargs (like output_hidden_states)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
