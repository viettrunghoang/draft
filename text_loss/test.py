import cv2
import torch
import numpy as np
from transformer import Transformer
from weight_ce_loss import weight_cross_entropy
from text_focus_loss import TextFocusLoss  


def load_image_as_tensor(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (128, 32))  
    img = img.astype(np.float32) / 255.0  
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  
    return img

img1 = load_image_as_tensor("AK5T06.jpg")
img2 = load_image_as_tensor("AK5T06.png")

args = {} 
# loss_fn = TextFocusLoss(args).cuda()
loss_fn = TextFocusLoss(args)

# labels = ["AK5T06", "AK5T06"]  
labels = ["kjajsca"]

loss_value = loss_fn(img1, img2, labels)
print("Loss value:", loss_value.item())
