import torch
import torchvision.transforms as transforms
from PIL import Image
from models.emnist_model import EmnistModel  # Import your model architecture

def preprocess_image(img):
    """
    Preprocess the input image for the EMNIST model.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure the image is grayscale
        transforms.Resize((28, 28)),  # Resize image to 28x28
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize based on training
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def load_model(model_path):
    """
    Load the pre-trained EMNIST model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmnistModel()  # Initialize the model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)  # Move model to device
    model.eval()  # Set to evaluation mode
    return model

def predict_image(img, model):
    """
    Predict the class of the input image using the specified model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)  # Move image to the same device as the model

    # Perform inference
    with torch.no_grad():
        outputs = model(img)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get class index with highest score
    return predicted.item()  # Return predicted class index
