import os
import gdown

MODEL_PATH = "model/basnet.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading basnet.pth...")
        url = "https://drive.google.com/uc?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu"
        os.makedirs("model", exist_ok=True)
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model already exists. Skipping download.")

# Run on import
download_model()
