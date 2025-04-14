from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

# Define the search keywords and the number of images to download for each
search_queries = ['blood', 'cloth', 'gun', 'fingerprint']

def download_images(query):
    arguments = {
        "keywords": query,
        "limit": 100,  # Number of images to download
        "print_urls": True,
        "output_directory": "C:/COLLEGE/web develop/Evidence_prediction/data/train",
        "format": "jpg",
        "no_directory": False
    }
    try:
        response.download(arguments)
    except Exception as e:
        print(f"Error downloading images for {query}: {e}")

for query in search_queries:
    download_images(query)
