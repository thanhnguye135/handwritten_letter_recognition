import os
from PIL import Image
import torchvision.transforms as transforms
from src.predict import predict_image

MODEL_PATH = "models/emnist_model.pth"

def main():
    # Load and preprocess test image
    img_path = "data/test_images/test_image.png"  # Replace with your image path
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform(img)
    
    # Predict character
    predicted_label = predict_image(img, MODEL_PATH)
    print(f"Predicted Character: {chr(predicted_label + 96)}")  # Convert to alphabet

if __name__ == "__main__":
    main()
