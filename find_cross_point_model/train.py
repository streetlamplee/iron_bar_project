import datetime
import gc
import os.path

import torch.cuda
from torch import optim, autocast
from torch.cpu.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from model import pointFindingModel
import extension
from data_processing import make_data
from dataset import PointDataset
from criterion import pointFinderCriterion
from EarlyStopping import EarlyStopping
import custom_transforms

print_with = extension.print_with
str_with = extension.str_with

def main():
    '''
    this func will use QPU section for train
    '''
    '''
    데이터 확인 및 없으면 만들기
    '''
    if not os.path.exists('./train/data.json'):
        make_data(f'warp_image', f'./data/train')
    if not os.path.exists('./valid/data.json'):
        make_data(f'warp_image', f'./data/valid')

    '''
    시드 고정
    '''
    extension.set_seed(42)

    '''
    학습에 적용할 augmentation
    '''
    train_transform = transforms.Compose([
        # transforms.RandomApply([
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomHorizontalFlip(),
        # ], 0.5),
        # transforms.RandomRotation(45, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # train_transform = custom_transforms.custom_transforms()

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    '''
    데이터 불러오기
    '''
    train_dataset = PointDataset('./train/data.json', train_transform)
    valid_dataset = PointDataset('./valid/data.json', valid_transform)

    '''
    데이터 로더 만들기
    '''
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=1, pin_memory=True, shuffle=False, drop_last=False)

    '''
    학습에 필요한 요소 선언
    '''
    num_classes = 1
    model = pointFindingModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = pointFinderCriterion()

    model.to(device)
    scaler = GradScaler()

    target_epochs = 99999
    min_val_loss = float('inf')
    es = EarlyStopping(patience=1999, mode='min', delta=1e-7)
    print_with(f'device: {device}')

    '''
    GPU 캐시 삭제
    '''
    torch.cuda.empty_cache()
    gc.collect()

    '''
    모델 학습
    '''
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_foldername = f'./models/{start_time}'
    if not os.path.exists(model_foldername):
        os.makedirs(model_foldername, exist_ok=True)
    model_list = []
    for epoch in range(1, target_epochs+1):
        model.train()
        train_loss = 0.0

        for image, target in tqdm(train_loader, desc=str_with(f"Epoch {epoch}/{target_epochs}")):
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output_logit = model(image)
                # output = torch.sigmoid(output_logit) # class가 1이 아닌 경우, softmax로 변경할 것
                # output = torch.sigmoid(output_logit)
                loss, _, _ = criterion(output_logit, target)

            if not torch.isfinite(loss):
                print_with("Loss NaN or Inf")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * image.size(0)
        epoch_loss = train_loss / len(train_dataset)


        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for v_image, v_target in tqdm(valid_loader, desc = str_with(f"Validation")):
                v_image, v_target = v_image.to(device), v_target.to(device)
                v_output_logit = model(v_image)
                # v_output = torch.sigmoid(v_output_logit)
                v_loss, _, _ = criterion(v_output_logit, v_target)

                valid_loss += v_loss.item() * v_image.size(0)

        valid_epoch_loss = valid_loss / len(valid_dataset)
        improved, early_stop = es.step(valid_epoch_loss)
        if early_stop:
            print_with("early stopped")
            break
        if improved:
            model_filename = os.path.join(model_foldername,f'epoch{epoch:05d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            },
            model_filename
            )
            model_list = os.listdir(model_foldername)
            if len(model_list) > 5:
                model_list.sort()
                oldest_model_filename = model_list[0]
                os.remove(os.path.join(model_foldername, oldest_model_filename))
            tqdm.write(f'{str_with("new checkpoint Saved")}')
        tqdm.write(f'{str_with(f"[{epoch}/{target_epochs}],Train Loss: {epoch_loss}, Loss: {valid_epoch_loss:.8f}")}')


if __name__ == '__main__':
    main()