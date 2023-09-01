import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_epoch(
    model: Module,
    optimizer: Optimizer,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device):


    # Loop through training data
    model.train()
    for img, labels in train_loader:
        
        # Load in batch and cast image to float32
        img = img.to(device) # (N,1,H,W)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(img) # (N,10), float32
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

    # Test model
    return test(model, val_loader, device)



def test(model: Module, val_loader: DataLoader, device):
    """ Returns performance metric on validation set """
    model.eval()
    n_correct_predictions = 0.0
    with torch.no_grad():
        for img, labels in val_loader:

            img, labels = img.to(device), labels.to(device)
            output_probability = model(img) # (N,10)

            predicted_batch_class = torch.argmax(output_probability, dim=-1) # (N,) class 0-9

            n_correct_predictions += (predicted_batch_class == labels).sum().cpu().item()

    accuracy = 100*(n_correct_predictions / len(val_loader.dataset))
    return accuracy