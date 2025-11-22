import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SegNet  # import class SegNet của bạn

# ----------------------------
# 1. Load model và checkpoint
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 22  # theo label_nbr của bạn
model = SegNet(input_nbr=3, label_nbr=n_classes)

checkpoint = torch.load(r"C:\Users\ADMIN\Documents\AI\DenseFusion\src\vanilla_segmentation\checkpoints\best_seg_model.pth",
                        map_location=device)

# Nếu train bằng DataParallel, loại bỏ 'module.'
state_dict = checkpoint['model_state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# ----------------------------
# 2. Load test image
# ----------------------------
img_path = r"C:\Users\ADMIN\Documents\AI\DenseFusion\dataset\data\0005\000001-color.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# Normalize & convert to tensor
img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0

# ----------------------------
# 3. Run segmentation
# ----------------------------
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]  # shape: (H, W)

# ----------------------------
# 4. Show results
# ----------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)

plt.subplot(1,2,2)
plt.title("Segmentation Mask")
plt.imshow(pred_mask, cmap='jet', alpha=0.7)
plt.show()
