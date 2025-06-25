from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import ast
import os


class MovieDataset(Dataset):
    def __init__(self, train=True, root="data"):
        self.root = root

        self.padding_token = 0
        if not os.path.exists(os.path.join(root, "train.pkl")) or not os.path.exists(os.path.join(root, "test.pkl")):
            print("Preparing data...")
            self._prepare_data()

        if train:
            self.df = pd.read_pickle(os.path.join(root, "train.pkl"))
        else:
            self.df = pd.read_pickle(os.path.join(root, "test.pkl"))
        
        self.num_labels = len(self.df["label"].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = row["review_tokens"]
        y = row["label"]
        att_mask = row["attention_mask"]
        return torch.tensor(x), torch.tensor(y), torch.tensor(att_mask)

    def _prepare_data(self):
        path = "dataPrep/data/movies_tokenized.csv"
        df = pd.read_csv(path)
        
        # Convert string representations to lists
        df["review_tokens"] = df["review_tokens"].apply(ast.literal_eval)
        df["attention_mask"] = df["attention_mask"].apply(ast.literal_eval)
        
        df = df[df["review_tokens"].apply(len) > 0]

        # Codify sentiments into integers
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["sentiment"]).astype(int)

        train_df, test_df = train_test_split(
            df, train_size=0.6, random_state=42, stratify=df["label"]
        )

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        train_df.to_pickle(os.path.join(self.root, "train.pkl"))
        test_df.to_pickle(os.path.join(self.root, "test.pkl"))