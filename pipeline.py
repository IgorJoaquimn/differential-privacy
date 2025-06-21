import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import DistilBertForSequenceClassification
from dataset import MovieDataset

# The functions are defined in the same order as they are usually run in the pipeline.


def set_seed(seed=42):
    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = (
        True  # Makes training slower but ensures reproducibility
    )
    torch.backends.cudnn.benchmark = False


def train(model, dataloader, optimizer, criterion, scheduler, device, num_epochs):
    losses = []
    for epoch in tqdm(range(num_epochs), desc="training epochs"):
        loss = train_epoch(model, dataloader, optimizer, criterion, scheduler, device)
        losses.append(loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    return losses


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0.0

    for x, y, att_mask in dataloader:
        x, y, att_mask = x.to(device), y.to(device), att_mask.to(device)
        logits = model(x, attention_mask=att_mask).logits
        loss = criterion(logits, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y, att_mask in dataloader:
            x, y, att_mask = x.to(device), y.to(device), att_mask.to(device)
            logits = model(x, attention_mask=att_mask)
            predictions = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f}, F1 Score (weighted): {f1:.4f}")

    return acc, f1


if __name__ == "__main__":
    # set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4

    dataset = MovieDataset(max_length=512, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=dataset.num_labels
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train(model, dataloader, optimizer, criterion, scheduler, device, num_epochs)

    dataset = MovieDataset(max_length=512, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate(model, dataloader, device)
