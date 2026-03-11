# 3D Face Reconstruction + Material Transfer 계획

## 목표
2D 얼굴 이미지 → 3D 복원 → 마스크 재질(silicone, resin, latex, plaster) 적용 → 동일 시점 2D 렌더링  
iBeta Level 2 인증을 위한 합성 3D 마스크 공격 데이터 생성

## 프로젝트 구조

```
face_reconstruction/
├── data/test_images/          # 입력 이미지 (CelebA-HQ 5장)
├── methods/
│   ├── Deep3DFaceRecon_pytorch/  # BFM 기반 3D 복원
│   └── emoca/                    # FLAME 기반 3D 복원
├── scripts/
│   ├── detect_landmarks.py       # MTCNN 5-point landmark 검출
│   ├── convert_and_compare.py    # .obj→.ply 변환 + 비교 그리드
│   └── render_deep3d.py          # (예정) Deep3D 재질 변환 렌더링
├── outputs/
│   ├── deep3d/                   # Deep3DFaceRecon 결과 (.obj, .png)
│   ├── emoca/                    # EMOCA 결과 (.obj, .png, .npy)
│   ├── ply/                      # MeshLab용 .ply 변환 파일
│   └── comparison_grid.png       # 두 방법 비교 이미지
└── docs/
    └── plan.md                   # 이 문서
```

## 완료된 단계

### Step 1: 2D → 3D 얼굴 복원

두 가지 방법을 설치하고 실행 완료:

| 항목 | Deep3DFaceRecon | EMOCA v2 |
|------|-----------------|----------|
| 기반 모델 | BFM (Basel Face Model) | FLAME |
| conda env | `deep3d` (Python 3.9) | `emoca` (Python 3.8) |
| PyTorch | 1.12.1+cu113 | 1.12.1+cu113 |
| 출력 | .obj (vertex color) + .mat (BFM 계수) | .obj (vertex color) + .npy (FLAME 파라미터) |
| 메시 범위 | 얼굴 전면만 (마스크 형태) | 머리 전체 (두피, 뒷통수 포함) |
| 상태 | 5장 inference 완료 | 5장 inference 완료 |

#### Deep3DFaceRecon .mat 계수 구조
- `id` (80): shape identity 계수
- `exp` (64): expression 계수
- `tex` (80): texture 계수
- `angle` (3): 회전 (Euler angles)
- `gamma` (27): Spherical Harmonics 조명 (3채널 x 9차)
- `trans` (3): translation
- `lm68` (68x2): 2D landmark 좌표

#### 렌더링 파라미터
- focal length: 1015
- image center: 112
- camera distance: 10
- projection: perspective
- crop size: 224x224

## 진행 중인 단계

### Step 2: 재질 변환 렌더링 (Deep3DFaceRecon 우선)

#### Step 2-1: 원본 시점 렌더링 정합 확인
- `.mat` 계수로 3D face 복원
- 원본 시점에서 2D 렌더링
- 원본 이미지 위에 overlay하여 정합 확인
- 스크립트: `scripts/render_deep3d.py`

#### Step 2-2: Silicone 재질 적용
- 실리콘 재질 특성:
  - 부드러운 표면 (고주파 디테일 감소)
  - 피부와 유사하나 약간 왁스질감
  - subsurface scattering 효과 (반투명)
  - 약간 탈채도, 균일한 피부톤
- 3D 노멀 + SH 조명을 활용한 재질 효과 적용
- 원본 이미지 위에 자연스럽게 합성

#### Step 2-3: 원본 이미지에 합성 (compositing)
- 메시 마스크로 얼굴 영역 추출
- soft blending으로 경계 자연스럽게 처리
- 머리카락, 배경 보존

### Step 3: 다른 재질 추가 (Silicone 완료 후)
- Resin (수지): 반투명, 광택
- Latex (라텍스): 고무질감, 약간 광택
- Plaster (석고): 무광, 하얀색

### Step 4: EMOCA 기반 동일 작업
- FLAME 파라미터 활용
- Deep3DFaceRecon 결과와 비교

### Step 5: 파이프라인 자동화
- 입력 이미지 → 3D 복원 → 재질 변환 → 2D 합성 이미지 one-shot 파이프라인
