import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    print(f"Downloading weights from: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as f, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

if __name__ == "__main__":
    # Standard Food-101 EfficientNetB0 Weights
    # Note: These weights are compatible with the EfficientNetB0 architecture
    WEIGHTS_URL = "https://github.com/chaze0/Food101-EfficientNetB0-Weights/releases/download/v1.0/food101_efficientnetb0_weights.h5"
    DESTINATION = "model.h5"
    
    if os.path.exists(DESTINATION):
        print(f"'{DESTINATION}' already exists. Skipping download.")
    else:
        try:
            download_file(WEIGHTS_URL, DESTINATION)
            print("\n✅ AI Brain (Weights) downloaded successfully!")
            print("Now restart your Streamlit app to see the AI in action.")
        except Exception as e:
            print(f"\n❌ Error downloading weights: {e}")
            print("Please check your internet connection or download the file manually.")
