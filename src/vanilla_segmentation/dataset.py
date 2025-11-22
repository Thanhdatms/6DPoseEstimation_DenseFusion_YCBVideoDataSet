import os
import random
import copy
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageEnhance, ImageFilter
import scipy.io as scio

import torch
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class SegmentationDataset(data.Dataset):
    def __init__(self, root_dir, txtlist, use_noise):
        """
        Args:
            root_path: root path contain full data sorce
            txtlist: txt file contain file list
            use_noise: turn on augumentation
            transform: 
        """
        self.root_dir = root_dir
        self.use_noise = use_noise
        # read file path
        with open(txtlist, 'r') as f:
            self.path = [line.strip() for line in f if line.strip()] # including real and synthetic

        self.real_path = [p for p in self.path if p.startswith('data/')] # only real data not synthetic

        self.data_len = len(self.path)
        self.back_len = len(self.real_path)

        # Augmentation
        self.color_aug = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        # Normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        

    def _load_image(self, subpath, suffix):
        if suffix == 'color':
            full = os.path.join(self.root_dir, f"{subpath}-{suffix}.jpg")
        else:
            full = os.path.join(self.root_dir, f"{subpath}-{suffix}.png")

        return Image.open(full)
    

    def __getitem__(self, idx):
        index = idx % self.data_len
        item = self.path[index]
        # load label + meta
        label = np.array(self._load_image(item, "label")) 
        meta = scio.loadmat(os.path.join(self.root_dir, f"{item}-meta.mat"))

        # load RGB
        rgb = self._load_image(item, "color").convert("RGB")
        if self.use_noise:
            rgb = self.color_aug(rgb)

        # Synthenic data augmentation
        if "syn" in item:
            # Enhance + blur
            rgb = ImageEnhance.Brightness(rgb).enhance(1.5) # tang sang
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=0.8)) # lam mo mau nhe
            rgb = self.color_aug(rgb)

            # random backgroud 
            # randomly choose one image
            # load backgroud + segemetation mask cua anh nen
            seed = random.randint(0, self.back_len - 1)
            back_item = self.real_path[seed]

            back = self.color_aug(
                self._load_image(back_item, "color").convert("RGB")
            )
            back_label = np.array(self._load_image(back_item, "label"))

            # foregroud = backgroud + synthetic
            # Masking
            mask = (label == 0)

            rgb_np = np.array(rgb, dtype=np.float32)
            back_np = np.array(back, dtype=np.float32)

            # Add gaussian noise
            rgb_np += np.random.normal(0, 5.0, rgb_np.shape)
            rgb_np = np.clip(rgb_np, 0, 255)

            # Compose 
            rgb_np[mask] = back_np[mask]
            label[mask] = back_label[mask]
        
        else:
            rgb = np.array(rgb)

        # Random flips (augmentation)
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                rgb = np.fliplr(rgb)
                label = np.fliplr(label)
            elif choice == 1:
                rgb = np.flipud(rgb)
                label = np.flipud(label)
            elif choice == 2:
                rgb = np.flipud(np.fliplr(rgb))
                label = np.flipud(np.fliplr(label))

        # Convert to tensors
        rgb = torch.tensor(rgb.transpose(2, 0, 1).copy(), dtype=torch.float32) / 255.0
        rgb = self.normalize(rgb)

        target = torch.tensor(label.copy(), dtype=torch.int64)
        return rgb, target

    def __len__(self):
        return self.data_len


def visualize_sample(rgb_tensor, label_tensor):
    """
    Convert tensor -> numpy and visualize.
    """
    # Tensor: (C, H, W)
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    label = label_tensor.cpu().numpy()

    # Undo normalization (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = rgb * std + mean
    rgb = np.clip(rgb, 0, 1)

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("RGB Image")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Label")
    plt.imshow(label, cmap="jet")
    plt.colorbar()
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
  
    # Path example - CẦN SỬA cho đúng với dataset của bạn
    root_dir = r"C:\Users\ADMIN\Documents\AI\DenseFusion\dataset\data"
    txt_list = r"C:\Users\ADMIN\Documents\AI\DenseFusion\YCB-Video-Base\image_sets\test_data_train.txt"

    # Create dataset
    dataset = SegmentationDataset(
        root_dir=root_dir,
        txtlist=txt_list,
        use_noise=True,
    )

    print("Dataset length:", dataset.__len__())
    # Dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Grab one batch
    print("Loading one sample...")
    rgb, label = next(iter(loader))

    print("rgb shape:", rgb.shape)     # [1, C, H, W]
    print("label shape:", label.shape) # [1, H, W]

    # Visualize sample
    visualize_sample(rgb[0], label[0])