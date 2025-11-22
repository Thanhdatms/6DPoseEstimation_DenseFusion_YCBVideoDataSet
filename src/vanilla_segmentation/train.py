import os
import time
import numpy as np
import argparse
import torch
from PIL import Image
import scipy.io as scio
from torch.utils.data import DataLoader

# from lib.utils import setup_logger, seed_everything
from src.vanilla_segmentation.dataset import SegmentationDataset
from src.vanilla_segmentation.model import SegNet
from src.vanilla_segmentation.loss import Loss
from src.lib.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root',
                    default=os.path.join(os.getcwd(), 'dataset', 'data'),
                    help='data root dir of YCB_Video')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--n_epochs', type=int, default=600, help='epochs to train')
parser.add_argument('--n_workers', type=int, default=10, help='number of workers')
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--logs_path', 
                    default=os.path.join(os.getcwd(), 'src', 'vanilla_segmentation', 'logs'), help='path to save logs')
parser.add_argument('--model_save_path', 
                    default=os.path.join(os.getcwd(), 'src', 'vanilla_segmentation', 'checkpoints'), help='save seg model')
parser.add_argument('--resume_model', default='', help='resume train model')
opt = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# seed_everything(42)


def main():
    # dataset and dataloader
    
    train_dataset = SegmentationDataset(opt.dataset_root, 'src/settings/dataset_config/train_list.txt', True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.n_workers))

    test_dataset = SegmentationDataset(opt.dataset_root, 'src/settings/dataset_config/val_list.txt', False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_workers)

    print("Train dataset length:", train_dataset.__len__())
    print("Test dataset length:", test_dataset.__len__())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # model, optimizer, criterion
    model = SegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = Loss()

    # Logger
    os.makedirs(opt.logs_path, exist_ok=True)
    os.makedirs(opt.model_save_path, exist_ok=True)
    logger = setup_logger('train_logger', os.path.join(opt.logs_path, 'train.log'))

    # Check resume model
    start_epoch = 1 # epoch default if not resume
    best_val_loss = np.inf # intial large loss -> update after

    if opt.resume_model:
        checkpoint_path = os.path.join(opt.model_save_path, opt.resume_model)
        checkpoint = torch.load(checkpoint_path, map_location=device) # read checkpoint from path
        model.load_state_dict(checkpoint['model_state_dict']) # load paramter
        optimizer.load_state_dict(checkpoint['model_state_dict']) # load optimizer status
        start_epoch = checkpoint.get('epoch', 1)
        best_val_loss = checkpoint.get('best_val_loss', np.inf)

    # traning loop
    st_time = time.time()

    for epoch in range(start_epoch, opt.n_epochs+1):
        model.train()
        train_loss = 0.0
        print(f"Starting epoch {epoch}...")
        for batch_idx, (images, seg_labels) in enumerate(train_dataloader):
            images = images.to(device)
            seg_labels = seg_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, seg_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch}/{opt.n_epochs}], Step [{batch_idx+1}/{len(train_dataset)}], Loss: {loss.item():.4f}")
        avg_train_loss = train_loss / len(train_dataset)
        logger.info(f"Epoch [{epoch}/{opt.n_epochs}] Training Loss: {avg_train_loss:.4f}")

        # validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, seg_labels) in enumerate(test_dataloader):
                images = images.to(device)
                seg_labels = seg_labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, seg_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_dataloader)
        logger.info(f"Epoch [{epoch}/{opt.n_epochs}] Validation Loss: {avg_val_loss:.4f}")

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(opt.model_save_path, f"best_seg_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, save_path)
            logger.info(f"Saved best model at epoch {epoch} with val loss {best_val_loss:.4f}")

    end_time = time.time()
    logger.info(f"Training completed in {(end_time - st_time)/60:.2f} minutes.")


if __name__ == '__main__':
    main()


