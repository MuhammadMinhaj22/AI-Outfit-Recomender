# download_models.py
import os
import urllib.request

HF_USERNAME = "MuhammadM22"
HF_DATASET = "outfit-recommender-files"
BASE_URL = f"https://huggingface.co/datasets/{HF_USERNAME}/{HF_DATASET}/resolve/main"

# List all files that are on HuggingFace
FILES = {
    "outfit_model.joblib": f"{BASE_URL}/outfit_model.joblib",
    "outfit_model.pkl": f"{BASE_URL}/outfit_model.pkl",
    "feedback.db": f"{BASE_URL}/feedback.db",
}

def download_files():
    for filename, url in FILES.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"{filename} downloaded successfully!")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping.")

if __name__ == "__main__":
    download_files()