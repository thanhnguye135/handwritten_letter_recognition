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
from src.utils import get_default_device, to_device, DeviceDataLoader
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

dataset = EMNIST(root='data/raw', split='letters', train=True, download=True, transform=transforms.ToTensor())
test_dataset = EMNIST(root='data/raw', split='letters', train=False, download=True, transform=transforms.ToTensor())

img_tensor, labels = dataset[0]
print(img_tensor.shape, labels-1)

img_tensor, labels = dataset[1]
print(img_tensor.shape, labels-1)

img_tensor, labels = dataset[2]
print(img_tensor, labels-1)

val_per = 0.1
val_len = int(val_per*len(dataset))
train_ds, val_ds = random_split(dataset, [len(dataset)-val_len, val_len])
print(len(train_ds), len(val_ds))

logger.info(f"Training samples: {len(train_ds)}")
logger.info(f"Validation samples: {len(val_ds)}")

# DataLoader for training and validation
batch_size = 128

train_load = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_load = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

device = get_default_device()
print(device)

train_load = DeviceDataLoader(train_load, device)
val_load = DeviceDataLoader(val_load, device)

input_size = 28*28
output_size = 26
# 2. Initialize the model
logger.info("Initializing the model...")

model = EmnistModel()
to_device(model, device)

for images, label in train_load:
    labels = to_device(torch.tensor([x-1 for x in label]), device)
    outputs = to_device(model(images), device)
    break

probs = F.softmax(outputs, dim=1)
max_probs, preds = torch.max(probs, dim=1)
print(preds)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

print(accuracy(outputs, labels))

loss_fn = F.cross_entropy

loss = loss_fn(outputs, labels)
print(loss)

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
     
history = [evaluate(model, val_load, device)]
print(history)

# 6. Train the model
logger.info("Starting training pipeline...")
epochs = 15
max_lr = 0.05
grad_clip = 0.5
weight_decay = 1e-3

history += fit_one_cycle(epochs, max_lr, model, train_load, val_load, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay)

# 7. Plot Accuracy
logger.info("Plotting accuracy curve...")
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

plot_losses(history)

len(test_dataset)
img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label-1)

img, label = test_dataset[1]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label-1)

img, label = test_dataset[3000]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label-1)

print(img.unsqueeze(0).shape)

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_dataset[20000]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted Label:', predict_image(img, model) + 1, ', Predicted Alphabet:', classes[predict_image(img, model)])

img, label = test_dataset[5000]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted Label:', predict_image(img, model) + 1, ', Predicted Alphabet:', classes[predict_image(img, model)])

test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size=128), device)
result = evaluate(model, test_loader)
result

# 8. Save the trained model
model_save_path = 'models/emnist_model.pth'
torch.save(model.state_dict(), model_save_path)
logger.info(f"Model saved to {model_save_path}")
