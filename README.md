# Weather-Based Men's Outfit Recommender

"This is a smart outfit recommendation application that helps users dress appropriately based on location. Simply enter the name of any city from around the world, and the app will suggest the perfect outfit tailored to that city's current weather and climate conditions."

Make `.env` file and type `OPENWEATHER_API_KEY=YOUR_API_KEY_HERE`

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
