import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

# Add the root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# Add src directory to sys.path
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

# Import your model and utility functions
from src.utils import accuracy
from models.emnist_model import EmnistModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 1. Load and preprocess the dataset
logger.info("Loading and preprocessing dataset...")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel (grayscale)
    transforms.Resize((28, 28)),                 # Match model's expected input size
    transforms.ToTensor(),                       # Convert to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))         # Normalize as per training
])

dataset = EMNIST(root='data/raw', split='letters', train=True, download=True, transform=transform)
test_dataset = EMNIST(root='data/raw', split='letters', train=False, download=True, transform=transform)

val_percent = 0.1
val_size = int(len(dataset) * val_percent)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

logger.info(f"Training samples: {len(train_ds)}")
logger.info(f"Validation samples: {len(val_ds)}")

# DataLoader for training and validation
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=128, pin_memory=True)

# 2. Initialize the model
logger.info("Initializing the model...")
model = EmnistModel().to(device)

# 3. Define evaluation function
def evaluate(model, val_loader):
    logger.info("Evaluating model on validation set...")
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            images, labels = [x.to(device) for x in batch]
            result = model.validation_step((images, labels))
            outputs.append(result)
    return model.validation_epoch_end(outputs)

# 4. Get learning rate function
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 5. Training loop
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    logger.info("Starting training...")
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        logger.info(f"Epoch [{epoch+1}/{epochs}] started...")
        model.train()
        train_losses = []
        lrs = []
        for i, batch in enumerate(train_loader):
            images, labels = [x.to(device) for x in batch]
            loss = model.training_step((images, labels))
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()

            if i % 10 == 0:  # Log every 10 steps
                logger.info(f"Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    logger.info("Training completed.")
    return history

# 6. Train the model
logger.info("Starting training pipeline...")
epochs = 15
max_lr = 0.05
grad_clip = 0.5
weight_decay = 1e-3

history = fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, grad_clip=grad_clip, weight_decay=weight_decay)

# 7. Plot Accuracy
logger.info("Plotting accuracy curve...")
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.show()

# 8. Save the trained model
model_save_path = 'models/emnist_model.pth'
torch.save(model.state_dict(), model_save_path)
logger.info(f"Model saved to {model_save_path}")
