import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# config
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd() 
download_path = './house_price_data'
os.makedirs(download_path, exist_ok=True)

api = KaggleApi()
api.authenticate()

# download
competition_name = 'house-prices-advanced-regression-techniques'
print(f"Downloading dataset from competition '{competition_name}'...")
api.competition_download_files(competition_name, path=download_path)
print("Download complete!")

# unzip
zip_path = os.path.join(download_path, f'{competition_name}.zip')
print("Unzipping...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)
print("Unzipped successfully!")

os.remove(zip_path)
print("ZIP file deleted.")