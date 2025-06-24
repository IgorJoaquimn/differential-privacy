import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class MovieDataset(Dataset):
    def __init__(self, train=True, max_length=128):
        self.padding_token = 0
        self.max_length = max_length
        self.tokens, self.labels, self.num_labels = self._prepare_data(train)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx][: self.max_length]
        y = self.labels[idx]
        att_mask = [1] * len(x)  # Attention mask for the tokens
        if len(x) < self.max_length:
            x = x + [self.padding_token] * (self.max_length - len(x))
            att_mask += [0] * (self.max_length - len(att_mask))
        return torch.tensor(x), torch.tensor(y), torch.tensor(att_mask)

    def _prepare_data(self, train):
        path = "dataPrep/data/movies_tokenized.csv"
        df = pd.read_csv(path)


        df["review_tokens"] = df["review_tokens"].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )
        df = df[df["review_tokens"].apply(len) > 0]

        # Codify sentiments into integers
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["sentiment"])
        num_labels = df["label"].nunique()

        tokens = df["review_tokens"].tolist()
        labels = df["label"].tolist()
        x_train, x_test, y_train, y_test = train_test_split(
            tokens, labels, train_size=0.6, random_state=42
        )

        if train:
            return x_train, y_train, num_labels
        return x_test, y_test, num_labels
