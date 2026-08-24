"""
철근 segmentation 모델 학습 스크립트.

실행: 이 파일이 있는 폴더(src/iron_bar_segmentation)에서 python train.py
      경로가 상대경로라서 다른 위치에서 실행하면 데이터를 찾지 못한다.

학습 데이터: data_real/data(원본 사진)와 data_real/mask(정답 마스크)를 이름순으로 짝지어 사용한다.
결과물: models/{시작시각}/epoch{번호}.pth 로 저장되며, 추론 시에는 가장 최근 파일이 자동 선택된다.
"""

import datetime
import gc
import json
import os.path
import time
import numpy as np
import torch.cuda
from torch import optim, autocast
from torch.cpu.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from criterion import BCELoss, DiceLoss
from extension import image_show
from model import DeepLabv3Plus
import extension
from dataset import ironbar_dataset
import criterion
from EarlyStopping import EarlyStopping
from custom_transform import custom_transforms
import random

print_with = extension.print_with
str_with = extension.str_with

def main():
    '''
    시드 고정
    같은 조건으로 다시 돌렸을 때 결과가 재현되도록 난수를 고정한다.
    '''
    extension.set_seed(42)

    '''
    학습에 적용할 augmentation
    '''
    train_transform = custom_transforms('train')

    valid_transform = custom_transforms('valid')

    '''
    데이터 불러오기
    사진과 마스크를 각각 이름순으로 정렬해 같은 순번끼리 짝짓는다.
    따라서 두 폴더의 파일 이름 규칙이 서로 맞아야 한다.
    '''
    image_list = os.listdir('data_real/data')
    mask_list = os.listdir('data_real/mask')
    image_list.sort()
    mask_list.sort()

    # 사진과 마스크 개수가 다르면 짝이 어긋나므로 즉시 중단한다.
    if len(image_list) != len(mask_list):
        raise 'len error'

    # 전체의 90%를 학습, 나머지 10%를 검증에 사용한다.
    train_idx = random.sample(range(len(image_list)), int(len(image_list) * 0.9))
    valid_idx = [idx for idx in range(len(image_list)) if idx not in train_idx]

    train_image_list = [os.path.join('data_real/data',image_list[i]) for i in train_idx]
    valid_image_list = [os.path.join('data_real/data',image_list[i]) for i in valid_idx]

    train_mask_list = [os.path.join('data_real/mask',mask_list[i]) for i in train_idx]
    valid_mask_list = [os.path.join('data_real/mask',mask_list[i]) for i in valid_idx]

    train_dataset = ironbar_dataset(train_image_list, train_mask_list, train_transform)
    valid_dataset = ironbar_dataset(valid_image_list, valid_mask_list, valid_transform)

    '''
    데이터 로더 만들기
    사진 해상도가 커서 batch_size를 작게 잡았다. drop_last: 마지막 자투리 배치는 버린다.
    '''
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2, num_workers=1, pin_memory=True, shuffle=False, drop_last=False)

    '''
    학습에 필요한 요소 선언
    '''
    # 클래스는 "철근" 하나뿐이므로 출력 채널 1개
    num_classes = 1
    model = DeepLabv3Plus(num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 픽셀 단위 정확도(BCE)와 영역 겹침(Dice)을 함께 본다.
    bce = BCELoss()
    dice = DiceLoss()

    model.to(device)
    # AMP(자동 혼합정밀도)용 scaler. 메모리를 아끼고 학습을 빠르게 한다.
    scaler = GradScaler()

    # 종료 시점은 epoch 수가 아니라 EarlyStopping이 결정한다.
    # 다만 patience가 999라 사실상 자동 종료되지 않으므로, 보통 직접 중단시킨다.
    target_epochs = 99999
    min_val_loss = float('inf')
    es = EarlyStopping(patience=999, mode='min', delta=1e-5)
    print_with(f'device: {device}')

    '''
    GPU 캐시 삭제
    '''
    torch.cuda.empty_cache()
    gc.collect()

    '''
    모델 학습
    '''
    # 실행 시각으로 폴더를 만들어, 이전 학습 결과를 덮어쓰지 않게 한다.
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
            # image_show(image[0].squeeze(0).permute(1,2,0).detach().cpu().numpy())
            # image_show(target[0].squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255)

            optimizer.zero_grad()
            # autocast: 연산 일부를 float16으로 수행해 메모리와 시간을 줄인다.
            with autocast(device_type='cuda'):
                output_logit = model(image)
                # output = torch.sigmoid(output_logit) # class가 1이 아닌 경우, softmax로 변경할 것
                # output = torch.sigmoid(output_logit)
                b = bce(output_logit, target)
                d = dice(output_logit, target)
                loss = b + d

            # float16 연산에서 값이 발산하면 loss가 NaN/Inf가 된다.
            # 이대로 역전파하면 모델 전체가 망가지므로 해당 배치를 건너뛴다.
            if not torch.isfinite(loss):
                print_with("Loss NaN or Inf")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 배치 크기가 다를 수 있으므로 장수를 곱해 더하고, 마지막에 전체 장수로 나눈다.
            train_loss += loss.item() * image.size(0)
        epoch_loss = train_loss / len(train_dataset)


        # 검증 단계: 학습에 쓰지 않은 데이터로 성능을 확인한다.
        # eval()로 BatchNorm/Dropout 동작을 바꾸고, no_grad로 기울기 계산을 끈다.
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for v_image, v_target in tqdm(valid_loader, desc = str_with(f"Validation")):
                v_image, v_target = v_image.to(device), v_target.to(device)
                v_output_logit = model(v_image)
                # v_output = torch.sigmoid(v_output_logit)
                v_b = bce(v_output_logit, v_target)
                v_d = dice(v_output_logit, v_target)
                v_loss = v_b + v_d

                valid_loss += v_loss.item() * v_image.size(0)

        valid_epoch_loss = valid_loss / len(valid_dataset)
        # 검증 loss가 좋아졌는지 확인해 저장 여부와 조기 종료 여부를 판단한다.
        improved, early_stop = es.step(valid_epoch_loss)
        if early_stop:
            print_with("early stopped")
            break
        # 기록을 갱신했을 때만 checkpoint를 저장한다.
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
            # 용량을 아끼기 위해 최신 5개만 남기고 가장 오래된 checkpoint를 지운다.
            model_list = os.listdir(model_foldername)
            if len(model_list) > 5:
                model_list.sort()
                oldest_model_filename = model_list[0]
                os.remove(os.path.join(model_foldername, oldest_model_filename))
            tqdm.write(f'{str_with("new checkpoint Saved")}')
        tqdm.write(f'{str_with(f"[{epoch}/{target_epochs}],Train Loss: {epoch_loss}, Loss: {valid_epoch_loss:.8f}")}')


if __name__ == '__main__':
    main()