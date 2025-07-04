import argparse
import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from dataset import MovieDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from models.baseline import BaselineModel
from models.madlib import MadlibModel
from models.tem import TEMModel

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
    torch.backends.cudnn.deterministic = True  # Makes training slower but ensures reproducibility
    torch.backends.cudnn.benchmark = False


def create_results_dir(num_epochs, batch_size, learning_rate, epsilon=None, folder="results"):
    """Create organized results directory structure"""
    if(folder!= "results"):
        results_dir = f"{folder}/results/{num_epochs}/{batch_size}/{learning_rate}"
        checkpoints_dir = f"{folder}/checkpoints/{num_epochs}/{batch_size}/{learning_rate}"
    else:
        results_dir = f"results/{num_epochs}/{batch_size}/{learning_rate}"
        checkpoints_dir = f"checkpoints/{num_epochs}/{batch_size}/{learning_rate}"
    if epsilon is not None:
        results_dir += f"/epsilon_{epsilon}"
        checkpoints_dir += f"/epsilon_{epsilon}"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    return results_dir, checkpoints_dir


def train(model, dataloader, optimizer, criterion, device, num_epochs, args, checkpoints_dir):
    losses = []
    accs = []

    for epoch in range(num_epochs):
        loss, acc = train_epoch(model, dataloader, optimizer, criterion, device)
        losses.append(loss)
        accs.append(acc)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            try:
                save_model(model, args, checkpoints_dir)  # Pass checkpoints_dir
            except:
                pass
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    return losses, accs


def train_with_privacy(
    original_model,
    dataloader,
    optimizer,
    criterion,
    device,
    num_epochs,
    args,
    checkpoints_dir,
    epsilon,
    delta,
    max_grad_norm=1.0,
    max_physical_batch_size=8,
):
    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=original_model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_delta=delta,
        target_epsilon=epsilon,
        epochs=num_epochs,
        max_grad_norm=max_grad_norm
    )

    losses = []
    accs = []
    epsilons = []
    
    for epoch in range(num_epochs):
        with BatchMemoryManager(
            data_loader=dataloader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_dataloader:
            loss, acc = train_epoch(model, memory_safe_dataloader, optimizer, criterion, device)
        losses.append(loss)
        accs.append(acc)

        curr_epsilon = privacy_engine.accountant.get_epsilon(delta)
        epsilons.append(curr_epsilon)

        if epoch % 5 == 0 or epoch == num_epochs - 1 or curr_epsilon > 5.0:
            try:
                save_model(original_model, args, checkpoints_dir)
            except:
                pass
            print(f"Privacy spent: epsilon={curr_epsilon:.4f}")
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Current epsilon: {curr_epsilon:.4f}, Accuracy: {acc:.4f}"
        )
    
    return losses, accs, epsilons


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for x, y, att_mask in tqdm(dataloader, desc="epoch"):
        x, y, att_mask = x.to(device), y.to(device), att_mask.to(device)

        optimizer.zero_grad()

        logits = model(x, attention_mask=att_mask)
        loss = criterion(logits, y.long())
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

        loss.backward()
        optimizer.step()

    acc = accuracy_score(y_true, y_pred)

    return total_loss / len(dataloader), acc


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y, att_mask in tqdm(dataloader):
            x, y, att_mask = x.to(device), y.to(device), att_mask.to(device)
            logits = model(x, attention_mask=att_mask)
            predictions = logits.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f}, F1 Score (weighted): {f1:.4f}")

    return acc, f1


def save_model(model, args, checkpoints_dir):
    # Create filename based on whether it's a checkpoint or final model
    base_name = f"private_{args.model}" if args.run_private else args.model
    fname = f"{checkpoints_dir}/{base_name}.pt"
    
    torch.save(model.state_dict(), fname)
    print(f"Model saved to {fname}")


def get_model(name, **kwargs):
    models = {
        "baseline": BaselineModel,
        "tem": TEMModel,
        "madlib": MadlibModel,
    }

    if name not in models:
        raise ValueError(
            f"Model {name} not supported. Options are: {list(models.keys())}"
        )

    return models[name](**kwargs)


if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train a movie review classifier")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "tem", "madlib"],
        default="baseline",
        help="Pretrained model name",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate the model instead of training")
    parser.add_argument("--run_private", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--target_epsilon", type=float, default=7.5, help="Target epsilon for differential privacy")
    parser.add_argument("--folder", type=str, default="results", help="Folder to save results")
    args = parser.parse_args()

    print(f"Executing with arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epsilon = args.target_epsilon

    dataset = MovieDataset(train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    delta = 1 / len(dataloader)  # Set delta based on the number of batches

    # Create organized directory structure
    results_dir, checkpoints_dir = create_results_dir(num_epochs, batch_size, learning_rate, epsilon, folder=args.folder)

    model = get_model(args.model, num_labels=dataset.num_labels, epsilon=epsilon).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    if args.eval:
        base_name = f"private_{args.model}" if args.run_private else args.model
        fname = f"{checkpoints_dir}/{base_name}.pt"
        model.load_state_dict(torch.load(fname, map_location=device))
    else:
        if args.run_private:
            losses = train_with_privacy(
                model,
                dataloader,
                optimizer,
                criterion,
                device,
                num_epochs,
                args,
                checkpoints_dir,
                epsilon,
                delta
            )
        else:
            losses = train(model, dataloader, optimizer, criterion, device, num_epochs, args, checkpoints_dir)

        # Save losses to text file in results directory
        fname = f"{results_dir}/{args.model}_losses.txt"
        if args.run_private:
            fname = f"{results_dir}/{args.model}_private_losses.txt"
        with open(fname, "w") as f:
            for loss in losses:
                f.write(f"{','.join(f'{val:.4f}' for val in loss)}\n")

    dataset = MovieDataset(train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    acc, f1 = evaluate(model, dataloader, device)

    # Save results in organized directory
    fname = f"{results_dir}/{args.model}_results.txt"
    if args.run_private:
        fname = f"{results_dir}/{args.model}_private_results.txt"
    with open(fname, "w") as f:
        f.write(f"Accuracy: {acc:.4f}, F1 Score (weighted): {f1:.4f}\n")
