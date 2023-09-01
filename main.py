import torch
import torch.nn as nn
from model import CNN
from functools import partial
from training import train_epoch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from checkpointer import TorchCheckpointer

# Simulation parameters 
batch_size = 64
lr = 0.001
n_epochs = 16
n_branches = 6
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# Initialize data
train_data = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Create model
model = CNN()
model = model.to(device)

# Optimizer and loss function 
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Trainable function
partial_kwargs = {
    "loss_fn": loss_fn,
    "train_loader": train_loader,
    "val_loader": test_loader,
    "device": device}

trainable = partial(
    train_epoch,
    **partial_kwargs
) 

# Create checkpointers
checkpointer = TorchCheckpointer(
    trainable=trainable,
    model=model,
    optimizer=optimizer,
    model_name="CNN",
    n_branches=n_branches
)

checkpointer.train(model=model, optimizer=optimizer, n_epochs=n_epochs)

checkpointer.tree.print_tree()

checkpointer.tree.print_3d_tree()