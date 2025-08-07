import os
import gdown
import torch
from basnet_model import BASNet  # ✅ Correct import only

def load_model():
    model_path = "basnet.pth"
    if not os.path.exists(model_path):
        print("Downloading BASNet model...")
        gdown.download(
            "https://drive.google.com/uc?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu",
            model_path,
            quiet=False
        )
    model = BASNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

