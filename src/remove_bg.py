import cv2
import numpy as np
from PIL import Image
import torch

from src.engine import normPRED
from src.dataset import img2tensor
from src.cfg import config


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (config.img_sz)) 
    img = img.astype(np.float32) / 255.0 
    return img


def get_predictions(img, model, device):
    img = img2tensor(img).unsqueeze(axis=0)
    model.eval()
    model.to(device)
    outputs = model(img.to(device))
    pred = outputs[5][:,0,:,:]
    pred = normPRED(pred)
    pred = pred.squeeze().detach().cpu().numpy()
    return pred

def remove_bg(image, result):
    result[result > 0.5] = 1
    result[result <= 0.5] = 0

    mask = Image.fromarray(result*255).convert('RGBA')
    back = Image.new("RGBA", (mask.size), (255, 255, 255))
    mask = mask.convert('L')
    img = Image.fromarray(np.uint8(image*255), 'RGB')
    im_out = Image.composite(img, back, mask)
    return im_out