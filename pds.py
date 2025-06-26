import argparse
import os

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import faiss

from models.madlib import MadlibModel
from models.tem import TEMModel 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, required=True, help="Epsilon value for the model")
    parser.add_argument("--model", type=str, choices=["madlib", "tem"], required=True, help="Which model to use")
    return parser.parse_args()


def get_model_and_tokenizer(model_name, epsilon):
    if model_name == "madlib":
        model = MadlibModel(num_labels=2, epsilon=epsilon)
    elif model_name == "tem":
        model = TEMModel(num_labels=2, epsilon=epsilon)
    else:
        raise ValueError("Invalid model")

    tokenizer = model.tokenizer
    return model, tokenizer


def collect_token_embeddings_in_batches(model, tokenizer, device, batch_size=32, num_repeats=1000):
    model.eval()
    token_ids = sorted(list(tokenizer.get_vocab().values()))
    idx = []
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(token_ids), batch_size), desc="Collecting token embeddings"):
            batch_token_ids = token_ids[i:i+batch_size]
            input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
            input_ids = input_ids.repeat_interleave(num_repeats).view(-1, 1)
            token_embeds = model.get_embeddings(input_ids).squeeze(1).cpu()

            for j, token_id in enumerate(batch_token_ids):
                start = j * num_repeats
                end = start + num_repeats
                idx.append(token_id)
                embeddings.append(token_embeds[start:end])

    return idx, embeddings



def compute_closest_embeddings_faiss(idx_list, embedding_list, embedding_matrix, tokenizer, batch_size=1024):
    """
    Computes the most similar tokens (from a reference embedding matrix) for a list of token embeddings using FAISS on GPU.

    Args:
        idx_list (list[int]): List of token IDs.
        embedding_list (list[Tensor]): List of token embedding tensors (one per token).
        embedding_matrix (Tensor): Tensor of shape (V, D) with reference embeddings.
        tokenizer: HuggingFace tokenizer.
        batch_size (int): Batch size for querying.

    Returns:
        pd.DataFrame with:
            - token_id
            - closest_token_id
            - similarity
            - token
            - closest_token
    """
    assert len(idx_list) == len(embedding_list), "idx_list and embedding_list length mismatch"

    # Normalize and convert embedding matrix to NumPy
    embedding_matrix = F.normalize(embedding_matrix, dim=1)
    embedding_dim = embedding_matrix.shape[1]
    embedding_matrix_np = embedding_matrix.cpu().detach().numpy()

    # Always use GPU
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(embedding_dim)
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index.add(embedding_matrix_np)

    all_closest_token_ids = []
    all_similarities = []

    # Process in batches
    for batch_start in tqdm(range(0, len(idx_list), batch_size), desc="FAISS search"):
        batch_end = min(batch_start + batch_size, len(idx_list))
        batch_embeddings = embedding_list[batch_start:batch_end]

        # Stack and normalize
        batch_tensor = torch.stack(batch_embeddings)  # (B, D)
        batch_tensor = F.normalize(batch_tensor, dim=1)

        # batch_tensor shape: (batch_size, 100, embed_size)
        B, N, D = batch_tensor.shape
        batch_tensor_reshaped = batch_tensor.reshape(B * N, D)  # (B*100, embed_size)
        batch_tensor_reshaped = F.normalize(batch_tensor_reshaped, dim=1)  # normalize all vectors

        batch_tensor_np = batch_tensor_reshaped.cpu().detach().numpy()

        indices,similarities  = index.search(batch_tensor_np, k=1)  # now shape (B*100, 1)

        # reshape back if needed
        similarities = similarities.reshape(B, N)
        indices = indices.reshape(B, N)


        all_closest_token_ids.extend(similarities.tolist())
        all_similarities.extend(indices.tolist())

    # Create DataFrame
    df_results = pd.DataFrame({
        "token_id": idx_list,
        "closest_token_id": all_closest_token_ids,
        "similarity": all_similarities
    })
    # Add token strings
    df_results["token"] = tokenizer.convert_ids_to_tokens(df_results["token_id"])
    df_results["closest_token"] = [tokenizer.convert_ids_to_tokens(x) for x in df_results["closest_token_id"]]

    return df_results


def main():
    args = parse_args()
    epsilon = args.epsilon
    model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model_and_tokenizer(model_name, epsilon)
    model.to(device).eval()
    embedding_matrix = model.original_emb.weight

    idx, embeddings = collect_token_embeddings_in_batches(
        model, tokenizer, device, batch_size=32, num_repeats=1000
    )

    df_results = compute_closest_embeddings_faiss(
        idx_list=idx,
        embedding_list=embeddings,
        embedding_matrix=embedding_matrix,
        tokenizer=tokenizer,
        batch_size=32
    )

    # Post-processing metrics
    df_results["nw"] = df_results.apply(lambda row: row["closest_token_id"].count(row["token_id"]), axis=1)
    df_results["sw"] = df_results["closest_token_id"].apply(lambda x: len(set(x)))


    # Save cumulative plot
    plt.figure(figsize=(8, 6))
    sns.histplot(
        df_results["sw"],
        bins=50,
        cumulative=True,
        stat="density",
        kde=False,
        color="steelblue"
    )
    plt.title(f"Cumulative Distribution of sw (ε = {epsilon})", fontsize=14)
    plt.xlabel("Number of Unique Closest Tokens", fontsize=12)
    plt.ylabel("Cumulative Proportion", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"figs/cumulative_sw_{model_name}_{epsilon}.pdf")
    plt.close()

    # Save DataFrame
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/closest_tokens_{model_name}_epsilon{epsilon}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")


if __name__ == "__main__":
    main()
