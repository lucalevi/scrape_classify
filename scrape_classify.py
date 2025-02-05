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
    """
    # Create the folder to save images
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Send a request to the URL
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch URL: {url}")
        return []

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")

    # List to hold paths of downloaded images
    image_paths = []

    # Download each image
    for i, img_tag in enumerate(img_tags):
        img_url = img_tag.get("src")
        if not img_url:
            continue

        # Handle relative URLs
        if not img_url.startswith("http"):
            img_url = requests.compat.urljoin(url, img_url)

        try:
            img_data = requests.get(img_url).content
            img_path = os.path.join(download_folder, f"image_{i}.jpg")
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            image_paths.append(img_path)
            print(f"Downloaded {img_url} to {img_path}")
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

    return image_paths


def classify_and_rename_images(image_paths, model, transform, download_folder="images"):
    """
    Classifies images and renames them based on their content.
    """
    # Load the labels for ImageNet
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = requests.get(labels_url).json()

    for img_path in image_paths:
        try:
            # Open the image
            image = Image.open(img_path).convert("RGB")

            # Apply transformations
            input_tensor = transform(image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_idx = outputs.max(1)
                label = labels[predicted_idx.item()]

            # Rename the file
            new_name = f"{label.replace(' ', '_').lower()}.jpg"
            new_path = os.path.join(download_folder, new_name)
            os.rename(img_path, new_path)
            print(f"Renamed {img_path} to {new_path}")
        except Exception as e:
            print(f"Failed to classify and rename {img_path}: {e}")


def main(url):
    # Step 1: Scrape and download images
    print("Scraping and downloading images...")
    download_folder = "images"
    image_paths = download_images(url, download_folder)

    if not image_paths:
        print("No images found.")
        return

    # Step 2: Load pre-trained model
    print("Loading pre-trained model...")
    model = models.resnet50(pretrained=True)
    model.eval()

    # Define the image transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Step 3: Classify and rename images
    print("Classifying and renaming images...")
    classify_and_rename_images(image_paths, model, transform, download_folder)


if __name__ == "__main__":
    # Provide the URL as input
    input_url = input("Enter the URL to scrape images from: ")
    main(input_url)
