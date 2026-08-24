# iron_bar_project

건설 현장에서 촬영한 **철근 사진**으로부터 철근을 segmentation 하고,
여러 시점의 사진을 하나의 평면으로 warping·합성하여
**특정 영역의 철근 배근 상태(상부근 / 하부근)** 를 확인하는 프로젝트입니다.

- **input** : 여러 각도에서 촬영한 철근 사진 (카메라 4대 기준)
- **output** : 철근 segmentation 결과, 관심 영역의 warping 이미지, 다중 시점 합성 결과 이미지

---

## 처리 파이프라인

`main.py`의 `run_manual_warp_pipeline(z)`가 현재 메인 실행 경로이며, 아래 순서로 동작합니다.

1. **데이터 로드** — `data/input_data/{target}/` 폴더의 사진들을 읽어옴
   (Raspberry Pi에서 직접 가져오거나, 동영상 프레임을 추출하는 경로도 준비되어 있음)
2. **철근 Segmentation** — DeepLabv3+ 모델로 철근 영역을 예측 (`src/iron_bar_segmentation/predict.py`)
3. **목표 구역 설정** — 관심 영역의 4점 좌표 지정
   - `FastDebug = 1` : `main.py`에 하드코딩된 target별 좌표 사용
   - `FastDebug = 0` : `PointClicker`로 이미지 위를 직접 클릭하여 좌표 입력
4. **z 보간** — `target == 1`인 경우, 상부근 좌표와 하부근 좌표를 `z`(0~100) 비율로 선형 보간하여
   높이별 단면을 생성 (`z=0` 하부근, `z=100` 상부근)
5. **Perspective Warp** — 지정한 4점을 1024×1024 평면으로 투영 (`src/processing/warp.py`)
6. **다중 시점 합성** — warping 된 segmentation 결과를 blur(dilate/erode/median/gaussian) 후
   가중 평균하고, 모든 시점에서 공통으로 검출된 영역만 threshold로 남김
7. **결과 저장** — `output/{target}/{z}/` 아래에 단계별 이미지 저장

### 출력 파일 규칙

| 파일 | 내용 |
| --- | --- |
| `1.{i}.png` | 원본 사진에 warp point를 표시한 확인용 이미지 |
| `2.{i}_seg.png` | 철근 segmentation 결과 |
| `3.{i}_warp.png` | 원본 사진의 warping 결과 |
| `4.{i}_seg_warp.png` | segmentation 결과의 warping 결과 |
| `5.result_before.png` | threshold 적용 전 다중 시점 가중 합성 결과 |
| `6.result.png` | threshold 적용 후 최종 결과 |
| `report_seg_pointed_{i}.png` | 보고용, segmentation 위에 warp point 표시 |

---

## 디렉터리 구조

```
.
├── main.py                     # 메인 실행 진입점 (수동 warp / SIFT warp 두 파이프라인)
├── requirements.txt
├── data/
│   ├── input_data/{target}/    # target 번호별 입력 사진
│   ├── data_sample/            # 테스트용 샘플 이미지
│   ├── chessboardImage/        # 카메라 캘리브레이션용 체스보드 이미지
│   └── video_frame/            # 동영상에서 추출한 프레임
├── output/{target}/{z}/        # 실행 결과 이미지
├── src/
│   ├── iron_bar_segmentation/  # 철근 segmentation 모델 (학습 / 추론)
│   │   ├── model.py            #   DeepLabv3+ (ResNet101 backbone)
│   │   ├── train.py            #   학습 스크립트
│   │   ├── predict.py          #   추론 (가장 최근 .pth를 자동 선택)
│   │   ├── dataset.py / criterion.py / custom_transform.py / EarlyStopping.py
│   │   ├── data_real/          #   학습용 실제 사진 및 마스크 생성 도구
│   │   └── models/{timestamp}/ #   학습된 checkpoint
│   ├── processing/             # 영상 처리 유틸
│   │   ├── warp.py             #   perspective transform
│   │   ├── blur.py             #   dilate/erode + median/gaussian blur
│   │   ├── pointClicker.py     #   마우스 클릭으로 4점 좌표 입력
│   │   ├── camera.py           #   체스보드 기반 카메라 캘리브레이션
│   │   ├── video_to_imageset.py#   동영상 → 프레임 추출
│   │   ├── draw_line.py / make_image_nice.py / brute_force_gpu.py / picture.py
│   ├── featureMatching/        # SIFT / SuperGlue 기반 특징점 매칭 및 Homography
│   ├── get/                    # Raspberry Pi 사진 수집, 3D→2D 좌표 투영
│   └── etc/                    # 공통 유틸 (image_show, seed 고정, cache manager 등)
└── runs/detect/                # YOLOv8 finetune 실험 결과 (참고용)
```

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`에 포함된 패키지: `numpy`, `opencv-python`, `matplotlib`, `tqdm`, `torch`, `torchvision`

> 참고: `src/get/get_picture_from_raspi.py`는 `paramiko`, `python-nmap`을 추가로 필요로 하며,
> 학습 스크립트(`train.py`)는 `Pillow`를 사용합니다.

---

## 실행

### 전체 파이프라인

```bash
python main.py
```

기본 설정은 `z`를 0부터 100까지 5 단위로 증가시키며 `run_manual_warp_pipeline(z)`를 반복 실행합니다.
실행 전에 `main.py` 상단의 아래 값들을 확인하세요.

- `target` : 처리할 입력 데이터 폴더 번호 (`data/input_data/{target}/`)
- `FastDebug` : `1`이면 하드코딩된 warp point 사용, `0`이면 클릭으로 직접 지정
- `isShow` : `1`이면 중간 결과를 창으로 표시 (GUI 환경 필요)

### 두 파이프라인의 차이

`main.py`에는 두 개의 파이프라인이 공존합니다. 현재 동작하는 경로는 `run_manual_warp_pipeline()`입니다.

| 항목 | `run_manual_warp_pipeline(z)` (현재 사용) | `run_sift_warp_pipeline()` (실험) |
| --- | --- | --- |
| warp point 획득 | target별 하드코딩 좌표 또는 `PointClicker` 수동 클릭 | 첫 장의 좌표를 기준으로 SIFT Homography로 나머지 장에 자동 전파 |
| 상/하부근 z 보간 | 지원 (`target == 1`) | 없음 |
| 입력 경로 | `data/input_data/{target}/` | `data/input_data/` |
| 카메라 대수 | 4대로 고정 (가중치 `1/4`, threshold `3/4`) | 입력 장수에 따라 동적 계산 |
| 결과 저장 | `output/{target}/{z}/` 아래 단계별 번호 파일 | 프로젝트 루트에 `weighted_sum_image.png` 등 2장 |
| 중간 결과 표시 | `isShow` 플래그로 제어 | 항상 `image_show()` 호출 (블로킹) |

두 함수의 핵심 계산(segmentation → warp → blur → 가중 합성 → threshold)은 동일하며,
차이는 **warp point를 어떻게 얻는가**와 **결과를 어떻게 정리하는가**에 있습니다.

### Segmentation 모델 학습

```bash
cd src/iron_bar_segmentation
python train.py
```

`data_real/data` (원본)와 `data_real/mask` (마스크)를 9:1로 나누어 학습하며,
checkpoint는 `models/{timestamp}/epoch{N}.pth`로 저장됩니다.
추론 시에는 `models/` 아래에서 **가장 최근에 수정된 `.pth`** 를 자동으로 선택합니다.

---

## 알려진 제약 / TODO

- `main.py`의 warp point가 target별로 하드코딩되어 있어, 새로운 데이터셋에는 클릭 입력이 필요합니다.
- 상부근/하부근 z 보간은 현재 `target == 1`에만 적용되어 있습니다.
- `run_sift_warp_pipeline()`(SIFT 기반 자동 warp point 전파)은 실험 단계이며, 현재 그대로는 실행되지 않습니다.
- 철근 교차점 검출 / 개수 카운팅 로직은 현재 파이프라인에서 비활성 상태입니다.
- 경로가 프로젝트 루트 기준 상대경로로 되어 있어, `main.py`는 루트에서 실행해야 합니다.
