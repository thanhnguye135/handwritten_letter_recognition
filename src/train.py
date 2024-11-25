import torch
from src.utils import to_device

def train_one_epoch(model, optimizer, train_loader, val_loader, device):
    """Train model for one epoch."""
    model.train()
    train_losses = []
    for batch in train_loader:
        images, labels = batch
        labels = labels - 1
        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(train_losses) / len(train_losses)

@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate the model."""
    model.eval()
    val_loss, val_acc = 0, 0
    for batch in val_loader:
        images, labels = batch
        labels = labels - 1
        outputs = model(images)
        val_loss += torch.nn.functional.cross_entropy(outputs, labels).item()
        val_acc += accuracy(outputs, labels)
    return {'val_loss': val_loss / len(val_loader), 'val_acc': val_acc / len(val_loader)}
