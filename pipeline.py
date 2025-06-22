import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForSequenceClassification
from dataset import MovieDataset
from opacus import PrivacyEngine

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
    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, scheduler, device)
        losses.append(loss)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            save_model(model, f"checkpoints/model_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    return losses # TODO: save losses somewhere


def train_with_privacy(
    model,
    dataloader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs,
    delta=1e-5,
):
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0, # TODO: adjust 
        max_grad_norm=1.0,
        batch_first=True
    )

    losses = []
    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, scheduler, device)
        losses.append(loss)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            save_model(model, f"checkpoints/private_model_epoch_{epoch + 1}.pt")
        curr_epsilon, best_alpha = privacy_engine.get_privacy_spent(delta)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Current epsilon: {curr_epsilon:.4f}, Best alpha: {best_alpha:.4f}")
    return losses


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0.0

    for x, y, att_mask in tqdm(dataloader, desc="epoch"):
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
            logits = model(x, attention_mask=att_mask).logits
            predictions = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f}, F1 Score (weighted): {f1:.4f}")

    return acc, f1


def save_model(model, path="model.pt"):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    batch_size = 8
    learning_rate = 1e-4

    dataset = MovieDataset(max_length=512, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=dataset.num_labels).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train(model, dataloader, optimizer, criterion, scheduler, device, num_epochs)

    dataset = MovieDataset(max_length=512, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate(model, dataloader, device)
