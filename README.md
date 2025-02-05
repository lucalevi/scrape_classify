# Image Scraper and Classifier

This project scrapes images from a given URL, downloads them, classifies them using a pre-trained ResNet-50 model, and renames them based on their predicted content.

## Requirements

Make sure you have Python 3.6 or higher installed. You can install the necessary libraries using `pip`:

```sh
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```
git clone https://github.com/lucalevi/scrape_classify.git
cd scrape_classify

```
2. Install the required libraries:
````
pip install -r requirements.txt
````

3.  Run the script:
````
python scrape_classify.py
````

4. Enter the URL to scrape images from when prompted.

### Example
```
Enter the URL to scrape images from: https://adastrafirst.wordpress.com/
```

The script will scrape images from the provided URL, download them to the images folder, classify them using the ResNet-50 model, and rename them based on their predicted content.

## Project Structure
- scrape_classify.py: The main script that performs scraping, downloading, classifying, and renaming of images.
- images: The folder where downloaded images are saved.

## License
This project is licensed under the MIT License.

