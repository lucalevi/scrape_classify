import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms


def download_images(url, download_folder="images"):
    """
    Scrapes and downloads all images from the given URL.

    Parameters:
    - url (str): The URL to scrape images from.
    - download_folder (str): The folder where downloaded images will be saved.

    Returns:
    - list: A list of file paths to the downloaded images.
    """
    # Create the folder to save images if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Send an HTTP GET request to the specified URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch URL: {url}")
        return []

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")  # Find all <img> tags

    # List to store file paths of downloaded images
    image_paths = []

    # Iterate over all <img> tags found
    for i, img_tag in enumerate(img_tags):
        img_url = img_tag.get("src")  # Extract the 'src' attribute
        if not img_url:
            continue  # Skip if 'src' attribute is missing

        # Handle relative URLs by converting them to absolute URLs
        if not img_url.startswith("http"):
            img_url = requests.compat.urljoin(url, img_url)

        try:
            # Download the image data
            img_data = requests.get(img_url).content
            # Define the file path for saving the image
            img_path = os.path.join(download_folder, f"image_{i}.jpg")
            # Save the image data to the file
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            image_paths.append(img_path)  # Add the path to the list
            print(f"Downloaded {img_url} to {img_path}")
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

    return image_paths


def classify_and_rename_images(image_paths, model, transform, download_folder="images"):
    """
    Classifies images using a pre-trained model and renames them based on their predicted content.

    Parameters:
    - image_paths (list): List of file paths to the images to classify.
    - model (torch.nn.Module): The pre-trained PyTorch model for image classification.
    - transform (torchvision.transforms.Compose): The transformation pipeline for preprocessing images.
    - download_folder (str): The folder where the images are stored.

    Returns:
    - None
    """
    # Download the ImageNet labels from the provided URL
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(labels_url).json()

    # Process each image in the list
    for img_path in image_paths:
        try:
            # Open the image using PIL and ensure it is in RGB format
            image = Image.open(img_path).convert("RGB")

            # Apply the transformation pipeline to prepare the image for model input
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Perform inference using the pre-trained model
            with torch.no_grad():
                outputs = model(input_tensor)  # Get model predictions
                _, predicted_idx = outputs.max(1)  # Get the index of the highest score
                label = labels[predicted_idx.item()]  # Map the index to the label

            # Rename the image file using the predicted label
            new_name = f"{label.replace(' ', '_').lower()}.jpg"  # Replace spaces with underscores
            new_path = os.path.join(
                download_folder, new_name
            )  # Define the new file path
            os.rename(img_path, new_path)  # Rename the file
            print(f"Renamed {img_path} to {new_path}")
        except Exception as e:
            print(f"Failed to classify and rename {img_path}: {e}")


def main(url):
    """
    Main function that orchestrates the process of scraping, downloading,
    classifying, and renaming images from a given URL.

    Parameters:
    - url (str): The URL to scrape images from.

    Returns:
    - None
    """
    # Step 1: Scrape and download images from the URL
    print("Scraping and downloading images...")
    download_folder = "images"  # Folder to save downloaded images
    image_paths = download_images(url, download_folder)

    # Check if there are any images downloaded
    if not image_paths:
        print("No images found.")
        return

    # Step 2: Load the pre-trained ResNet-50 model
    print("Loading pre-trained model...")
    model = models.resnet50(pretrained=True)  # Load ResNet-50
    model.eval()  # Set the model to evaluation mode

    # Define the transformation pipeline for image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225],  # Normalize with ImageNet std deviation
            ),
        ]
    )

    # Step 3: Classify and rename images
    print("Classifying and renaming images...")
    classify_and_rename_images(image_paths, model, transform, download_folder)


if __name__ == "__main__":
    """
    Entry point for the script. Prompts the user for a URL and runs the main function.
    """
    # Prompt the user to input a URL to scrape images from
    input_url = input("Enter the URL to scrape images from: ")
    main(input_url)
