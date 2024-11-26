import os
from PIL import Image
import matplotlib.pyplot as plt
from src.predict import preprocess_image, load_model, predict_image

# Define paths
MODEL_PATH = "models/emnist_model.pth"  # Path to your model file
IMG_PATH = "data/test_images/c.png"  # Path to the test image

def main():
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    if not os.path.exists(IMG_PATH):
        print(f"Image file not found: {IMG_PATH}")
        return

    # Load the image
    img = Image.open(IMG_PATH).convert("L")  # Ensure it's a grayscale image

    # Display the input image
    plt.imshow(img, cmap="gray")
    plt.title("Input Image")
    plt.show()

    # Preprocess the image
    img_tensor = preprocess_image(img)

    # Load the model
    model = load_model(MODEL_PATH)

    # Predict the class
    try:
        predicted_label = predict_image(img_tensor, model)
        predicted_character = chr(predicted_label + 96)  # Convert to alphabet (1=a, 2=b, ...)
        print(f"Predicted Label: {predicted_label}, Predicted Character: {predicted_character}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
