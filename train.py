import argparse
import numpy as np
from PIL import Image, ImageDraw
import os
import torch
from torch import autocast
from torch.amp import GradScaler
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import torch.optim as optim
import gc
from tqdm import tqdm
import datetime
import matplotlib
import matplotlib.pyplot as plt
# from DataProcessing import compute_mean_std
from torchvision.transforms.functional import InterpolationMode
import random
import model_architecture
import warnings
import threading
from torchvision.transforms import functional as F
import torchvision.tv_tensors
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToPILImage
import cv2
# import wandb
# from wandb.apis.importers.internals.util import for_each
matplotlib.use('TkAgg')

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed: int = 42):
    random.seed(seed)  # Python random seed 설정
    np.random.seed(seed)  # NumPy random seed 설정
    torch.manual_seed(seed)  # PyTorch CPU 시드 설정
    torch.cuda.manual_seed(seed)  # PyTorch GPU 시드 설정 (한 개의 GPU 사용 시)
    torch.cuda.manual_seed_all(seed)  # PyTorch 다중 GPU 사용 시 모든 GPU에 같은 seed 설정
    torch.backends.cudnn.deterministic = True  # CuDNN deterministic 설정
    torch.backends.cudnn.benchmark = False  # 성능보다 재현성을 우선할 경우 False로 설정

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지와 마스크 파일 경로
        original_idx = idx % len(self.image_files)
        image_name = self.image_files[original_idx]
        image_path = os.path.join(self.image_dir, image_name)
        # mask_name = image_name.replace(".jpg", ".png")  # 마스크 파일 이름
        # mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_name = image_name
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 이미지와 마스크 로드
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 마스크는 Grayscale로 로드
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        if self.transform :
            image, mask = self.transform(image, mask)
            Normalization = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            image = Normalization(image)

        image = image.float()
        mask = (mask).long()

        return image, mask

# wandb.login()
# run = wandb.init(project = "cscam_iron_bar_sample")
# config = wandb.config


def show_image_plt(title, image):
    plt.figure(title)
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    # plt.show(block=False)
    plt.pause(2)


if __name__ == "__main__":
    plt.ion()
    os.makedirs("models", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", type=str)
    parser.add_argument("--model", type=str, help="Choose Model to Train")
    args = parser.parse_args()

    img_size = 512

    set_seed(42)

    # 사용 예시
    img_dir = 'data/train/image'  # 철근 사진 폴더 경로
    # mean, std = compute_mean_std(img_dir)
    # COCO/Imagenet 스타일 데이터에 흔히 쓰이는 값
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 이미지와 마스크에 각각 다른 전처리 변환 적용
    train_transform = v2.Compose([
            v2.Resize((img_size, img_size), InterpolationMode.NEAREST),
            v2.RandomApply([
            v2.RandomHorizontalFlip(p=1.0),
            v2.RandomVerticalFlip(p=1.0),
            v2.RandomRotation(90, interpolation=InterpolationMode.NEAREST)
        ], p=0.75),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ])

    test_transform = v2.Compose([
        # A.RandomCrop(height=img_size, width=img_size),
        # A.Resize(img_size, img_size, interpolation=1),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize((img_size, img_size), InterpolationMode.NEAREST),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    ])



    # 데이터셋과 데이터로더 생성
    image_dir = "data/train/image"
    mask_dir = "data/train/mask"
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform= train_transform)
    # sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset) * 10)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=1, pin_memory=True, shuffle=True, drop_last = True)
    # data_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler)

    # test_image_dir = "/content/data/test"
    # test_mask_dir = "/content/data_masked/test"
    valid_image_dir = "data/valid/image"
    valid_mask_dir = "data/valid/mask"

    valid_dataset = SegmentationDataset(image_dir=valid_image_dir, mask_dir=valid_mask_dir, transform= test_transform)
    # valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples = len(valid_dataset) * 1)
    valid_data_loader = DataLoader(valid_dataset, batch_size=8, num_workers=1, pin_memory=True, drop_last = False)
    # valid_data_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], sampler=valid_sampler)

    num_classes = 2
    model = model_architecture.get_model(num_classes, norm = args.norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 손실 함수 및 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # criterion = FocalLoss()
    criterion = torch.nn.CrossEntropyLoss()
    bceloss = torch.nn.BCEWithLogitsLoss()
    crossEntropy = torch.nn.CrossEntropyLoss()
    focal_loss = model_architecture.FocalLoss()
    dice_loss = model_architecture.DiceLoss()

    if args.model != None:
        model = model_architecture.get_model(num_classes, norm = args.norm)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict = False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    # GPU 사용 설정
    model.to(device)
    scaler = GradScaler()

    torch.cuda.empty_cache()
    gc.collect()

    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(f"models/{start_time}")

    num_epochs = 33333
    min_val_loss = float('inf')

    torch.cuda.empty_cache()
    model.to(device)

    es = 0
    print(f"[train.py] device: {device}")
    image_show_flag3 = True

    for epoch in range(num_epochs):
        if es == 1500:
            print(f"[train.py] Train Early Stopped")
            break

        image_show_flag = False
        image_show_flag2 = True

        if epoch < 5:
            image_show_flag = True

        model.train()
        running_train_loss = 0.0

        for images, masks in tqdm(data_loader, desc=f"[train.py] Epoch {epoch+1}/{num_epochs}"):
            # # train data 확인용 코드 -------------------------------------------------------------------------
            # if image_show_flag & image_show_flag2:
            #     ToPILImage()(images[0]).show()
            #     ToPILImage()(masks[0].type(torch.float)).show()
            #     image_show_flag2 = False
            #     print(np.max(masks.cpu().detach().numpy()))
            #     print(np.min(masks.cpu().detach().numpy()))
            #     print(masks.dtype)
            #     print(tmp.cnt_not_zero_one(masks))
            # # ----------------------------------------------------------------------------------------------

            images = images.to(device)
            masks = masks.to(device)
            masks = masks.squeeze(1)
            masks_onehot = torch.nn.functional.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                # Forward pass
                outputs_logit = model(images)
                if num_classes == 1:
                    outputs = torch.sigmoid(outputs_logit)
                else:
                    outputs = torch.softmax(outputs_logit,1)
                # 손실 계산
                # focal = focal_loss(outputs, masks)
                loss1 = dice_loss(outputs, masks_onehot)
                if num_classes == 1:
                    loss2 = bceloss(outputs_logit.squeeze(1), masks_onehot.float())
                else:
                    loss2 = crossEntropy(outputs_logit, masks)
                # loss = criterion(outputs, masks)
                loss = loss1 + loss2
                # loss = focal + dice

            if not torch.isfinite(loss):
                print("Loss NaN or Inf")
                continue

            # Backward and optimize
            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()


            running_train_loss += loss.item() * images.size(0)

        epoch_loss = running_train_loss / len(dataset)
        # scheduler.step()
        model.eval()
        running_valid_loss = 0.0

        image_show_flag = False
        image_show_flag2 = True
        if epoch < 1 or epoch % 100 == 0:
            image_show_flag = True

        with torch.no_grad():
            for v_images, v_masks in tqdm(valid_data_loader, desc=f"[train.py] Validation"):
                if image_show_flag3:
                    show_image_plt(f"valid Data", (np.transpose(v_images.numpy()[0], (1,2,0)) * 255).astype(np.uint8))
                    show_image_plt(f"Masks", (v_masks.squeeze(1).numpy()[0] * 127).astype(np.uint8))

                    image_show_flag3 = False

                v_images, v_masks = v_images.to(device), v_masks.to(device)
                v_masks = v_masks.squeeze(1)
                v_masks_onehot = torch.nn.functional.one_hot(v_masks, num_classes=num_classes).permute(0,3,1,2).float()
                v_outputs_logit = model(v_images)
                if num_classes == 1:
                    v_outputs = torch.sigmoid(v_outputs_logit)
                else:
                    v_outputs = torch.softmax(v_outputs_logit, 1)
                # valid_focal_loss = focal_loss(v_outputs, v_masks)
                if image_show_flag and image_show_flag2:
                    show_image_plt(f"epoch{epoch+1}", (torch.argmax(v_outputs, dim=1).cpu().detach().squeeze().numpy()[0] * 127).astype(np.uint8))

                    image_show_flag2 = False
                valid_loss1 = dice_loss(v_outputs, v_masks_onehot)
                if num_classes == 1:
                    valid_loss2 = bceloss(v_outputs_logit.squeeze(1), v_masks_onehot.float())
                else:
                    valid_loss2 = crossEntropy(v_outputs_logit, v_masks_onehot)
                # valid_loss = valid_focal_loss + valid_dice_loss
                valid_loss = valid_loss1 + valid_loss2

                running_valid_loss += valid_loss.item() * v_images.size(0)

        val_loss = running_valid_loss / len(valid_dataset)

        if val_loss < min_val_loss or (epoch+1) % 10 == 0:
            isNewMinvalid = ""
            if val_loss < min_val_loss:
                es = 0
                isNewMinvalid = "_best"
                min_val_loss = val_loss
            else: es += 1
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, f"models/{start_time}/epoch{epoch+1}{isNewMinvalid}.pth")
            tqdm.write(f"[train.py] train mean : {mean}, train std: {std}")
            tqdm.write(f"[train.py] [{epoch + 1}/{num_epochs}],Train Loss: {epoch_loss}, Loss: {val_loss:.8f}")
            tqdm.write(f"[train.py] New Checkpoint Saved {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            es += 1
            tqdm.write(f"[train.py] train mean : {mean}, train std: {std}")
            tqdm.write(f"[train.py] [{epoch + 1}/{num_epochs}],Train Loss: {epoch_loss}, Loss: {val_loss:.8f}")
        # wandb.log({
        #     "train_loss": epoch_loss,
        #     "train_bce": loss2,
        #     "train_dice": loss1,
        #     "valid_loss": val_loss,
        #     "valid_bce": valid_loss2,
        #     "valid_dice": valid_loss1,
        #     "lr": optimizer.param_groups[0]['lr'],
        #     "es": es
        # }, step= epoch)


