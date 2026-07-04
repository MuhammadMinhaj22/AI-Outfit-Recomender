# Outfit Recommender - HuggingFace Large File Upload Task

## Your Task
This project contains `.pkl`, `.joblib`, and database files that cannot be uploaded to GitHub because they are too large. Your job is to:

1. Upload all heavy files to HuggingFace
2. Create an auto-download script in the code
3. Update `.gitignore`
4. Push cleanly to GitHub

---

## Step 1: Scan the Project

First scan the entire folder and note:
- Which `.pkl` files exist and where
- Which `.joblib` files exist
- Any `.db` / `.sqlite` files
- Main entry point file (app.py / main.py / etc.)

```bash
find . -name "*.pkl" -o -name "*.joblib" -o -name "*.db" -o -name "*.sqlite" | head -30
```

---

## Step 2: Ask User to Create HuggingFace Dataset

Tell the user to do this manually in the browser:
1. Login at https://huggingface.co
2. Create "New Dataset" — name it: `outfit-recommender-files`
3. Visibility: **Public**
4. Upload all heavy files there
5. Copy each file's link — format will be:
   ```
   https://huggingface.co/datasets/USERNAME/outfit-recommender-files/resolve/main/FILENAME.pkl
   ```

**Ask the user:** "What is your HuggingFace username and which files did you upload?"

---

## Step 3: Create download_models.py

Once the user provides username and filenames, create this file in the project root:

```python
# download_models.py
import os
import urllib.request

HF_USERNAME = "YOUR_USERNAME_HERE"
HF_DATASET = "outfit-recommender-files"
BASE_URL = f"https://huggingface.co/datasets/{HF_USERNAME}/{HF_DATASET}/resolve/main"

# List all files that are on HuggingFace
FILES = {
    "model.pkl": f"{BASE_URL}/model.pkl",
    # "encoder.joblib": f"{BASE_URL}/encoder.joblib",
    # "data.db": f"{BASE_URL}/data.db",
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
```

**Important:** Only add filenames to the FILES dictionary that the user actually uploaded to HuggingFace.

---

## Step 4: Add to Top of Main File

Add these lines at the very top of the main entry file (app.py or main.py):

```python
from download_models import download_files
download_files()
```

---

## Step 5: Update .gitignore

Add these lines to `.gitignore` (create the file if it does not exist):

```
# Large ML files - will be downloaded from HuggingFace
*.pkl
*.joblib
*.db
*.sqlite

# Python cache
__pycache__/
*.pyc
.env
```

---

## Step 6: Push to GitHub

```bash
# Remove heavy files from git tracking if already tracked
git rm --cached *.pkl 2>/dev/null || true
git rm --cached *.joblib 2>/dev/null || true
git rm --cached *.db 2>/dev/null || true

# Add everything
git add .
git commit -m "feat: move large files to HuggingFace, add auto-download script"
git push origin main
```

---

## Step 7: Test

```bash
# Delete heavy files locally
# Then run the script to verify they download correctly
python download_models.py
```

---

## Important Notes

- Confirm with the user after each step that everything worked correctly
- If any file on HuggingFace is set to private, a token will also be required
- If the entry point is not app.py, add the download call to whatever the correct file is
- Also add this to README.md so others know what to do:
  ```
  ## Setup
  Run `python download_models.py` before starting the app.
  ```
