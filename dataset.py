import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class MovieDataset(Dataset):
    def __init__(self, train=True, max_length=256):
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

        # Extracts only the first genre or 'Unknown' if no genres are present
        df["genre"] = (
            df["genres"]
            .fillna("[]")
            .apply(lambda x: eval(x)[0]["name"] if eval(x) else "Unknown")
        )

        df["overview_tokens"] = df["overview_tokens"].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )
        df = df[df["overview_tokens"].apply(len) > 0]

        # Codify genres into integers
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["genre"])
        num_labels = df["label"].nunique()

        tokens = df["overview_tokens"].tolist()
        labels = df["label"].tolist()
        x_train, x_test, y_train, y_test = train_test_split(
            tokens, labels, test_size=0.6, random_state=42
        )

        if train:
            return x_train, y_train, num_labels
        return x_test, y_test, num_labels
