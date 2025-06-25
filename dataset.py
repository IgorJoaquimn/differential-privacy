import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class MovieDataset(Dataset):
    def __init__(self, train=True):
        self.padding_token = 0
        self.tokens, self.labels, self.attention_masks, self.num_labels = self._prepare_data(train)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx]
        y = self.labels[idx]
        att_mask = self.attention_masks[idx]
        return torch.tensor(x), torch.tensor(y), torch.tensor(att_mask)

    def _prepare_data(self, train):
        path = "dataPrep/data/movies_tokenized.csv"
        self.df = pd.read_csv(path)
        self.df = self.df[self.df["review_tokens"].apply(len) > 0]

        # Convert string representations of lists to actual lists
        self.df["review_tokens"] = self.df["review_tokens"].apply(ast.literal_eval)
        self.df["attention_mask"] = self.df["attention_mask"].apply(ast.literal_eval)

        # Codify sentiments into integers
        le = LabelEncoder()
        self.df["label"] = le.fit_transform(self.df["sentiment"])
        num_labels = self.df["label"].nunique()

        tokens = self.df["review_tokens"].tolist()
        labels = self.df["label"].tolist()
        attention_masks = self.df["attention_mask"].tolist()

        x_train, x_test, y_train, y_test, att_train, att_test = train_test_split(
            tokens, labels, attention_masks, train_size=0.6, random_state=42
        )

        if train:
            return x_train, y_train, att_train, num_labels
        return x_test, y_test, att_test, num_labels