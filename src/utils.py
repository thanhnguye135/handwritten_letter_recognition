import torch

def get_default_device():
    """Pick GPU if available, else CPU."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap DataLoader to move data to a device."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)

def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)  # Get the index of the max log-probability
    correct = (preds == labels).float()  # Check correct predictions
    acc = correct.sum() / len(correct)  # Average accuracy
    return acc
