"""
SIFT 대신 딥러닝 기반 특징점 매칭(SuperPoint + SuperGlue)을 써보려던 스케치.

주의: 이 파일은 그대로 실행되지 않는다.
  - SuperPoint/SuperGlue 모델 코드(models 패키지)가 이 저장소에 없다.
  - image0_gray, image1_gray 가 정의되어 있지 않다.
참고용으로만 남겨둔 코드다.
"""

import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) SuperPoint 특징점 추출기 (SIFT를 대체하는 역할)
superpoint = SuperPoint({'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}).to(device)

# 2) SuperGlue 매칭기 (FLANN + ratio test를 대체하는 역할)
superglue = SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).to(device)

# 3) 이미지 불러오기 & 특징점 추출
img0 = torch.from_numpy(image0_gray)[None, None].float().to(device) / 255.
img1 = torch.from_numpy(image1_gray)[None, None].float().to(device) / 255.

pred0 = superpoint({'image': img0})
pred1 = superpoint({'image': img1})

# 4) SuperGlue 매칭
pred = superglue({'image0': img0, **pred0,
                  'image1': img1, **pred1})

matches0 = pred['matches0']  # img0의 각 점이 img1의 어느 점과 매칭됐는지
conf = pred['matching_scores0']
