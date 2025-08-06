import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os

from download_model import load_model

model = load_model()
model.eval()

def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((320, 320))
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 255.0
    image_np = image_np.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    return image_tensor

def postprocess(prediction, original_size):
    pred = prediction.squeeze().cpu().data.numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_LINEAR)
    mask = pred > 128
    return mask

def remove(input_path, output_path):
    original = Image.open(input_path).convert('RGB')
    image_tensor = preprocess(input_path)
    
    with torch.no_grad():
        d_output = model(image_tensor)[0]  # Use the first output only
    
    mask = postprocess(d_output, original.size)
    rgba_image = original.convert("RGBA")
    data = rgba_image.getdata()
    
    new_data = []
    for item, m in zip(data, mask.flatten()):
        if m:
            new_data.append(item)
        else:
            new_data.append((255, 255, 255, 0))  # transparent
    
    rgba_image.putdata(new_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rgba_image.save(output_path)
