import torch
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) SuperPoint 특징점 추출기
superpoint = SuperPoint({'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024}).to(device)

# 2) SuperGlue 매칭기
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
