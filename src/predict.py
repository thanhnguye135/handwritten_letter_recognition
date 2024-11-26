import torch
import torchvision.transforms as transforms
from PIL import ImageOps
from models.emnist_model import EmnistModel  # Ensure this path matches your project structure

def preprocess_image(img):
    """
    Preprocess the input image for the EMNIST model.
    """
    print("Inverting the image...")
    img = ImageOps.invert(img)  # Invert the image to match training data

    # Log image size before resizing
    print(f"Image size before resizing: {img.size}")

    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure the image is grayscale
        transforms.Resize((28, 28)),  # Resize image to 28x28
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Match EMNIST normalization
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    print(f"Image tensor shape after transformation: {img_tensor.shape}")
    return img_tensor

def load_model(model_path):
    """
    Load the pre-trained EMNIST model.
    """
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmnistModel()  # Initialize the model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)  # Move model to device
    model.eval()  # Set to evaluation mode
    return model

def predict_image(img_tensor, model):
    """
    Predict the class of the input image using the specified model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)  # Move image to the same device as the model

    # Perform inference
    with torch.no_grad():
        print("Performing forward pass...")
        outputs = model(img_tensor)  # Forward pass
        _, predicted = torch.max(outputs, dim=1)  # Get class index with highest score

    # Log the output of the model (raw logits)
    print(f"Model output (logits): {outputs}")
    print(f"Predicted class index: {predicted.item()}")

    return predicted.item()  # Return predicted class index
