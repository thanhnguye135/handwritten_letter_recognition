import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.transforms as transforms


# Add the root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# Add src directory to sys.path
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(SRC_DIR)

from src.predict import preprocess_image, load_model, predict_image

# Define paths
MODEL_PATH = "models/emnist_model.pth"  # Path to your model file
IMG_PATH = "data/test_images/b1.png"  # Path to the test image

def main():
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    if not os.path.exists(IMG_PATH):
        print(f"Image file not found: {IMG_PATH}")
        return

    # Load the image
    print(f"Loading image from {IMG_PATH}...")
    img = Image.open(IMG_PATH).convert("L")  # Ensure it's a grayscale image

    # Preprocess the image
    print("Preprocessing the image...")
    processed_img = preprocess_image(img)  # Process the image

    # Log processed image shape
    print(f"Processed image tensor shape: {processed_img.shape}")

    # Convert processed tensor back to PIL Image for saving
    processed_img_pil = transforms.ToPILImage()(processed_img.squeeze(0))  # Remove batch dimension and convert to PIL Image

    # Display the processed image
    print("Displaying processed image...")
    plt.imshow(processed_img_pil, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")  # Hide axes for cleaner image display
    plt.show()

    # Create output folder if it doesn't exist
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique name for the file using current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"processed_{timestamp}.png")

    # Save the processed image
    processed_img_pil.save(output_path)
    print(f"Saved processed image to {output_path}")

    # Load the model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    # Predict the class
    try:
        print("Making prediction...")
        predicted_label = predict_image(processed_img, model)
        predicted_character = chr(predicted_label + 96)  # 1 = a, 2 = b, ...
        print(f"Predicted Label: {predicted_label}, Predicted Character: {predicted_character.upper()}")

        # Optionally, add prediction result to the processed image
        processed_img_pil = processed_img_pil.convert("RGB")  # Convert to RGB to allow text drawing
        draw = ImageDraw.Draw(processed_img_pil)
        font = ImageFont.load_default()  # Use default font (can be replaced with a specific font if needed)
        text = f"Predicted: {predicted_character.upper()}"
        text_position = (10, 10)  # You can adjust this position as needed
        draw.text(text_position, text, font=font, fill="white")  # Draw text on image

        # Save the image with prediction result
        output_with_prediction_path = os.path.join(output_folder, f"predicted_{timestamp}.png")
        processed_img_pil.save(output_with_prediction_path)
        print(f"Saved predicted processed image to {output_with_prediction_path}")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
