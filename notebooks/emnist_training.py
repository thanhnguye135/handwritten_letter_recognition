# Add the models directory to the system path
import sys
import os

# Add the root directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now try importing your model
from models.emnist_model import EmnistModel


import torch
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import random_split, DataLoader
from src.utils import to_device, DeviceDataLoader, accuracy
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 1. Load and preprocess the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the EMNIST dataset (letters split)
dataset = EMNIST(root='data/raw', split='letters', train=True, download=True, transform=transform)
test_dataset = EMNIST(root='data/raw', split='letters', train=False, download=True, transform=transform)

# Split the dataset into training and validation sets
val_percent = 0.1
val_size = int(len(dataset) * val_percent)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f'Training samples: {len(train_ds)}')
print(f'Validation samples: {len(val_ds)}')

# DataLoader for training and validation
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

# Move data to the chosen device
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

# 2. Initialize the model
model = EmnistModel()
model = to_device(model, device)

# 3. Training and evaluation functions
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            sched.step()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# 4. Train the model
epochs = 15
max_lr = 0.05
grad_clip = 0.5
weight_decay = 1e-3

history = fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, grad_clip=grad_clip, weight_decay=weight_decay)

# 5. Plot accuracy
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')

# 6. Save the trained model
torch.save(model.state_dict(), 'models/emnist_model.pth')
