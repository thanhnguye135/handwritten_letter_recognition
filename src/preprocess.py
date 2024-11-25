from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

def prepare_datasets(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = EMNIST(root='data/raw', split='letters', train=True, download=True, transform=transform)
    val_size = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
    return train_ds, val_ds
