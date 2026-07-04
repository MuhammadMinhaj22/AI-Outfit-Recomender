# Weather-Based Men's Outfit Recommender

This application recommends outfits based on current weather conditions.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the download script to fetch models and database:
   ```bash
   python download_models.py
   ```

3. Start the application:
   ```bash
   uvicorn main:app --reload
   ```
