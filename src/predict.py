import torch
from models.emnist_model import EmnistModel
from src.utils import to_device, get_default_device

def predict_image(img, model_path):
    device = get_default_device()
    model = EmnistModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = to_device(model, device)
    model.eval()
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item() + 1
