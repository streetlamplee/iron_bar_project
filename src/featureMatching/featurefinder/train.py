"""
철근 교차점 검출 모델 학습 스크립트.

iron_bar_segmentation/train.py 와 전체 흐름은 같고, 다루는 정답의 형태가 다르다.
여기서는 마스크 이미지가 아니라 "점 좌표 목록"을 격자 텐서로 바꿔 학습한다.

실행: 이 파일이 있는 폴더에서 실행해야 한다 (경로가 상대경로).
"""

import datetime
import gc
import json
import os.path
import cv2
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
import custom_transform
from train_valid_split import train_valid_split

print_with = extension.print_with
str_with = extension.str_with

def main():
    '''
    this func will use QPU section for train
    '''
    '''
    데이터 확인 및 없으면 만들기
    라벨 정보를 담은 data.json 이 없으면 원본 이미지에서 새로 만든다.
    '''
    if not os.path.exists('./data/data.json'):
        make_data(f'warp_image', f'./data')


    '''
    시드 고정
    '''
    extension.set_seed(42)

    '''
    학습에 적용할 augmentation
    '''

    train_transform = custom_transform.custom_transforms('train')

    valid_transform = custom_transform.custom_transforms('valid')

    '''
    데이터 불러오기
    '''
    # 85%를 학습, 15%를 검증에 사용
    train_dict, valid_dict = train_valid_split('./data/data.json', 0.85, 0.15)

    # 입력 해상도 512 -> 격자는 512/32 = 16칸이 된다.
    train_dataset = PointDataset(train_dict, train_transform, 512)
    valid_dataset = PointDataset(valid_dict, valid_transform, 512)

    '''
    데이터 로더 만들기
    '''
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False, drop_last=False)

    '''
    학습에 필요한 요소 선언
    '''
    model = pointFindingModel()
    # 아래 두 줄의 주석을 풀면 이전 checkpoint에서 이어서 학습할 수 있다.
    # checkpoint = torch.load('/home/user/PycharmProjects/iron_bar_sample_project/find_cross_point_model/models/20250624_161155/epoch00905.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = pointFinderCriterion()

    model.to(device)
    scaler = GradScaler()

    target_epochs = 99999
    min_val_loss = float('inf')
    # patience가 매우 커서 사실상 자동 종료되지 않는다. 보통 직접 중단시킨다.
    es = EarlyStopping(patience=5000, mode='min', delta=1e-4)
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
    # 학습 시작 직후 한 번만, 데이터와 정답이 제대로 짝지어졌는지 창으로 확인한다.
    # GUI가 없는 환경에서는 False로 두어야 한다.
    image_show_flag = True
    for epoch in range(1, target_epochs+1):
        model.train()
        train_loss = 0.0

        for image, target in tqdm(train_loader, desc=str_with(f"Epoch {epoch}/{target_epochs}")):
            if image_show_flag:
                image_copy = image[0].permute(1,2,0).numpy().copy()
                target_copy = target[0].permute(1,2,0).numpy().copy()

                image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
                target_copy = cv2.cvtColor(target_copy, cv2.COLOR_RGB2BGR)

                extension.image_show(image_copy)
                extension.image_show(target_copy)
                image_show_flag = False

            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            # autocast: 일부 연산을 float16으로 처리해 메모리와 시간을 아낀다.
            with autocast(device_type='cuda'):
                output_logit = model(image)
                # output = torch.sigmoid(output_logit) # class가 1이 아닌 경우, softmax로 변경할 것
                # output = torch.sigmoid(output_logit)
                # criterion은 (전체 loss, 좌표 loss, 존재여부 loss)를 돌려준다.
                # 여기서는 전체 loss만 사용한다.
                loss, _, _ = criterion(output_logit, target)

            # float16 연산이 발산해 NaN/Inf가 되면 그 배치는 건너뛴다 (모델이 망가지는 것을 방지).
            if not torch.isfinite(loss):
                print_with("Loss NaN or Inf")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * image.size(0)
        epoch_loss = train_loss / len(train_dataset)


        # 검증: 학습에 쓰지 않은 데이터로 성능을 확인한다.
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
        # 검증 loss가 좋아졌을 때만 checkpoint를 남긴다.
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
            # 용량 관리를 위해 최신 5개만 남긴다.
            model_list = os.listdir(model_foldername)
            if len(model_list) > 5:
                model_list.sort()
                oldest_model_filename = model_list[0]
                os.remove(os.path.join(model_foldername, oldest_model_filename))
            tqdm.write(f'{str_with("new checkpoint Saved")}')
        tqdm.write(f'{str_with(f"[{epoch}/{target_epochs}],Train Loss: {epoch_loss}, Loss: {valid_epoch_loss:.8f}")}')


if __name__ == '__main__':
    main()