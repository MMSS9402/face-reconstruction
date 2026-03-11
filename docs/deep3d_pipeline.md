# Deep3DFaceRecon 기반 3D 마스크 합성 파이프라인

> 최종 업데이트: 2026-03-09
> 목적: iBeta Level 2 인증을 위한 합성 3D 마스크 공격 데이터 생성

---

## 1. 프로젝트 목표

2D 얼굴 이미지로부터 3D 얼굴을 복원하고, 실리콘/레진/라텍스/석고 등의 마스크 재질을 적용한 뒤,
동일 시점의 2D 이미지로 다시 합성하여 **가짜 3D 마스크 공격 데이터**를 자동 생성하는 것.

---

## 2. 전체 아키텍처

```
data/test_images/              scripts/                        outputs/
  celeba_hq_00000.jpg    ──→  detect_landmarks.py        ──→  data/test_images/detections/
  celeba_hq_00001.jpg              │                              celeba_hq_00000.txt (5-point landmarks)
  ...                              ▼
                           Deep3DFaceRecon inference     ──→  (내부) checkpoints/.../epoch_20_000000/
                            (conda env: deep3d)                    celeba_hq_00000.mat (BFM 계수)
                                   │                               celeba_hq_00000.obj (3D 메시)
                                   ▼                               celeba_hq_00000.png (렌더링)
                           convert_and_compare.py        ──→  outputs/deep3d/    (복사된 .obj/.png/.ply)
                                   │                          outputs/ply/        (MeshLab용 .ply)
                                   ▼
                           render_deep3d.py              ──→  outputs/deep3d_render/   (BFM 렌더링 overlay)
                                   │                          outputs/deep3d_silicone/ (구버전 실리콘)
                                   ▼
                           material_transfer.py          ──→  outputs/deep3d_material/ (최종 실리콘 합성)
```

---

## 3. 환경 설정

| 항목 | 값 |
|------|-----|
| Conda 환경 | `deep3d` (Python 3.9) |
| PyTorch | 1.12.1+cu113 |
| 핵심 라이브러리 | numpy, scipy, opencv-python, scikit-image, Pillow |
| GPU | nvdiffrast (Deep3DFaceRecon inference 시 사용) |

---

## 4. 단계별 상세 설명

### 4.1. 입력 데이터

**위치**: `data/test_images/`

| 파일 | 설명 |
|------|------|
| `celeba_hq_00000.jpg` ~ `celeba_hq_00004.jpg` | CelebA-HQ에서 선택한 512x512 얼굴 이미지 5장 |
| `detections/celeba_hq_00000.txt` | MTCNN으로 검출한 5-point landmark (양쪽 눈, 코, 양쪽 입꼬리) |

5-point landmark 파일 형식:
```
x1 y1    # 왼쪽 눈
x2 y2    # 오른쪽 눈
x3 y3    # 코
x4 y4    # 왼쪽 입꼬리
x5 y5    # 오른쪽 입꼬리
```

### 4.2. Step 1: 랜드마크 검출 (`scripts/detect_landmarks.py`)

```bash
conda run -n deep3d python scripts/detect_landmarks.py
```

- MTCNN을 사용하여 `data/test_images/*.jpg`에서 얼굴 검출 + 5-point landmark 추출
- 결과를 `data/test_images/detections/` 폴더에 `.txt` 파일로 저장
- 이 landmark는 Deep3DFaceRecon의 전처리(얼굴 정렬 + 224x224 crop)에 필요

### 4.3. Step 2: Deep3DFaceRecon Inference

```bash
cd methods/Deep3DFaceRecon_pytorch
python test.py --name face_recon_feat0.2_augment --epoch 20 \
    --img_folder ../../data/test_images
```

**입력**: `data/test_images/*.jpg` + `detections/*.txt`

**처리 과정**:
1. 5-point landmark로 얼굴 정렬 (SimilarityTransform) → 224x224 crop
2. CNN이 crop된 이미지에서 BFM 파라미터 회귀
3. BFM 파라미터로 3D 메시 복원 + 렌더링

**출력 위치**: `methods/Deep3DFaceRecon_pytorch/checkpoints/face_recon_feat0.2_augment/results/test_images/epoch_20_000000/`

| 파일 | 형식 | 설명 |
|------|------|------|
| `*.mat` | MATLAB | **핵심 출력**. BFM 계수 (shape, expression, texture, pose, lighting, landmarks) |
| `*.obj` | Wavefront OBJ | 3D 메시 + vertex color (MeshLab에서 직접 못 열림, vertex color 형식 비호환) |
| `*.png` | PNG | Deep3DFaceRecon이 자체 렌더링한 224x224 이미지 |

#### .mat 파일 내부 구조

| 키 | Shape | 설명 |
|----|-------|------|
| `id` | (1, 80) | Shape identity 계수. BFM의 id_base와 곱하여 개인 얼굴 형상 생성 |
| `exp` | (1, 64) | Expression 계수. BFM의 exp_base와 곱하여 표정 변형 생성 |
| `tex` | (1, 80) | Texture 계수. BFM의 tex_base와 곱하여 피부색/텍스처 생성 |
| `angle` | (1, 3) | Euler 회전각 (pitch, yaw, roll). 3D 메시의 머리 방향 |
| `gamma` | (1, 27) | Spherical Harmonics 조명 (3채널 x 9차). 환경 조명 근사 |
| `trans` | (1, 3) | 3D 공간에서의 평행이동 (x, y, z) |
| `lm68` | (1, 68, 2) | 복원된 3D 메시를 2D에 투영한 68-point landmark 좌표 (224x224 crop 공간) |

#### BFM (Basel Face Model) 설명

BFM은 약 200명의 3D 스캔 얼굴에서 PCA로 학습한 parametric face model이다.

```
3D_shape = mean_shape + id_base × id_coeff + exp_base × exp_coeff
3D_texture = mean_texture + tex_base × tex_coeff
```

- `mean_shape`: 평균 얼굴 형상 (35,709 vertices × 3)
- `id_base`: identity 변형 기저 벡터 80개
- `exp_base`: expression 변형 기저 벡터 64개
- `face_buf`: 삼각형 면 인덱스 (70,789 faces × 3)

#### 카메라 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| focal length | 1015 | 투영 행렬의 초점 거리 |
| image center | 112 | 224/2. 투영 중심 |
| camera distance | 10 | 카메라와 얼굴 사이 거리 |
| projection | perspective | 원근 투영 |
| crop size | 224×224 | Deep3DFaceRecon 표준 크기 |

### 4.4. Step 3: 3D 메시 변환 및 비교 (`scripts/convert_and_compare.py`)

```bash
conda run -n deep3d python scripts/convert_and_compare.py
```

Deep3DFaceRecon의 `.obj` 파일은 vertex color를 `v x y z r g b` 형태로 저장하는데,
MeshLab의 OBJ 파서가 이를 인식하지 못한다 (Assertion 에러 발생).

이 스크립트는:
1. `.obj` → `.ply` 변환 (PLY는 vertex color 지원)
2. float 색상값 (0~1) → 0~255 int로 변환
3. EMOCA 결과도 동일하게 변환
4. 비교 grid 이미지 생성

**출력**:

| 폴더 | 내용 |
|------|------|
| `outputs/deep3d/` | Deep3DFaceRecon 원본 출력 복사 (.obj, .png, .ply) |
| `outputs/ply/` | MeshLab에서 열 수 있는 .ply 파일 (Deep3D + EMOCA 모두) |

MeshLab으로 3D 확인:
```bash
meshlab outputs/ply/deep3d_celeba_hq_00000.ply
```

### 4.5. Step 4: BFM 렌더링 정합 확인 (`scripts/render_deep3d.py`)

```bash
conda run -n deep3d python scripts/render_deep3d.py
```

.mat 계수로부터 3D 얼굴을 다시 복원하고, 원본 시점에서 2D로 렌더링하여 원본 이미지와 overlay 비교하는 스크립트.

**파이프라인 (이 스크립트 내부)**:
1. `.mat` → BFM 계수 로드
2. `mean_shape + id_base × id + exp_base × exp` → 3D vertices
3. Euler angles → 회전 행렬 → 변환 적용
4. perspective projection → 2D 좌표
5. SH lighting 적용 → vertex color
6. numpy rasterizer (z-buffer) → 224×224 렌더링
7. SimilarityTransform으로 원본 이미지 좌표로 역변환
8. 원본 이미지 위에 합성

**출력**: `outputs/deep3d_render/` — `Original | Overlay | Render` 3컬럼 비교 이미지

> 이 단계의 목적: 3D 복원이 원본 이미지와 잘 정렬되는지 시각적으로 확인

### 4.6. Step 5: 실리콘 재질 합성 (`scripts/material_transfer.py`) ← 최종 결과물

```bash
conda run -n deep3d python scripts/material_transfer.py
```

이전 단계(render_deep3d.py)에서는 BFM 텍스처를 사용하여 렌더링했으나,
BFM 텍스처가 저차원(80 coefficients)이라 원본 얼굴의 디테일을 잃어버리는 문제가 있었다.

**핵심 변경: 원본 이미지 보존 방식**

3D 복원에서 **기하 정보(노멀맵, 뎁스맵, 마스크)만** 추출하고,
재질 효과는 **원본 이미지 픽셀 위에 직접** 적용한다.

```
.mat 계수 → 3D 메시 복원 → 노멀맵 + 뎁스맵 + 마스크 래스터라이즈 (224×224)
                                          ↓
                               warpAffine (→ 원본 이미지 좌표)
                                          ↓
                     원본 이미지 + 노멀맵/뎁스맵 → 실리콘 재질 효과 적용
                                          ↓
                               눈/입 구멍 제외 (lm68 기반)
                                          ↓
                          소프트 블렌딩 → 최종 합성 이미지
```

#### 실리콘 재질 효과 상세 (apply_silicone 함수)

실제 실리콘 마스크의 시각적 특성을 기반으로 설계:

| 단계 | 효과 | 파라미터 | 설명 |
|------|------|---------|------|
| 1 | Surface smoothing | GaussianBlur(15,15) + bilateralFilter(9,75,75), blend 85% | 모공/주름/미세 디테일 제거. 실리콘의 매끈한 표면 재현 |
| 2 | Skin tone 균일화 | GaussianBlur(31,31) 기반 local deviation 70% 감소 | 혈류로 인한 미세 색상 변이 제거. 실리콘은 균일한 색 |
| 3 | 탈채도 (desaturation) | 65% 원본 + 35% grayscale | 실리콘의 바랜 색감 재현 |
| 4 | 왁스질 색조 (color cast) | R×1.06, G×1.02, B×0.88 | 노르스름하고 생기 없는 "죽은 피부" 느낌 |
| 5 | 대비 감소 | 70% 원본 + 30% 평균값 | 실리콘의 균일한 albedo |
| 6 | 스페큘러 반사 | broad(n^4 × 0.12) + sharp(n^20 × 0.08) | 3D 노멀 기반 플라스틱 같은 광택. 왁스질 sheen |
| 7 | 밝기 증가 | ×1.05 | 실리콘이 피부보다 빛 반사를 더 많이 함 |

#### 눈/입 구멍 처리

실제 실리콘 마스크는 눈구멍과 입구멍이 뚫려있어 착용자의 눈과 입이 보인다.

- lm68 랜드마크를 원본 이미지 좌표로 변환 (`tform(lm68)`)
- 왼쪽 눈 (landmark 36-41): 중심 기준 1.4배 확장
- 오른쪽 눈 (landmark 42-47): 중심 기준 1.4배 확장
- 입 (landmark 48-59): 중심 기준 1.2배 확장
- `cv2.fillPoly`로 구멍 마스크 생성 → GaussianBlur로 경계 부드럽게
- 얼굴 마스크에서 차감하여 해당 영역은 원본 유지

**출력**: `outputs/deep3d_material/` — `Original | Silicone` 2컬럼 비교 이미지

---

## 5. outputs 폴더 구조 정리

```
outputs/
├── deep3d/                     # Deep3DFaceRecon 원본 출력
│   ├── celeba_hq_00000.obj     #   3D 메시 (vertex color 포함)
│   ├── celeba_hq_00000.png     #   Deep3D 자체 렌더링 (224×224)
│   └── celeba_hq_00000.ply     #   MeshLab용 변환 파일
│
├── emoca/                      # EMOCA v2 원본 출력
│   └── EMOCA_v2_lr_mse_20/
│       └── celeba_hq_0000000/
│           ├── mesh_coarse.obj     # FLAME 기반 3D 메시
│           ├── shape.npy, exp.npy  # FLAME 파라미터
│           ├── geometry_coarse.png # geometry 시각화
│           └── out_im_coarse.png   # 렌더링 결과
│
├── ply/                        # MeshLab 호환 .ply 파일
│   ├── deep3d_celeba_hq_00000.ply    # Deep3D (얼굴 전면만, ~35K vertices)
│   └── emoca_celeba_hq_0000000.ply   # EMOCA (머리 전체, ~59K vertices)
│
├── deep3d_render/              # BFM 텍스처 렌더링 + 원본 overlay (정합 확인용)
│   └── celeba_hq_00000_overlay.png   # [Original | Overlay | Render]
│
├── deep3d_silicone/            # (구버전) BFM 텍스처 기반 실리콘 — 사용하지 않음
│   └── celeba_hq_00000_overlay.png
│
├── deep3d_material/            # ★ 최종 결과물: 원본 이미지 기반 실리콘 합성
│   └── celeba_hq_00000_silicone.png  # [Original | Silicone]
│
└── comparison_grid.png         # Deep3D vs EMOCA 비교 grid
```

---

## 6. Deep3DFaceRecon vs EMOCA 비교

두 가지 3D 복원 방법을 설치하고 테스트했다.

| 항목 | Deep3DFaceRecon | EMOCA v2 |
|------|-----------------|----------|
| 기반 모델 | BFM (Basel Face Model) | FLAME |
| conda env | `deep3d` (Python 3.9) | `emoca` (Python 3.8) |
| PyTorch | 1.12.1+cu113 | 1.12.1+cu113 |
| vertex 수 | ~35,709 | ~59,000+ |
| 출력 형식 | .obj + .mat (BFM 계수) | .obj + .npy (FLAME 파라미터) |
| 메시 범위 | **얼굴 전면만** (마스크 형태) | **머리 전체** (두피, 뒷통수 포함) |
| 특징 | 기하 정확도 높음 | 표정 포착력 좋음, 뒷통수에 앞면 텍스처 복사되는 문제 |

현재 material transfer는 **Deep3DFaceRecon 기반**으로 구현되어 있다.
Deep3DFaceRecon의 .mat 계수에서 기하 정보만 추출하여 사용.

---

## 7. 핵심 기술 요소

### 7.1. Numpy 래스터라이저

PyTorch나 OpenGL 없이 순수 numpy로 구현한 삼각형 래스터라이저.

- **Z-buffer**: 깊이 비교로 올바른 가림(occlusion) 처리 (작은 Z = 카메라에 가까움)
- **Barycentric interpolation**: 삼각형 내부 픽셀의 노멀/뎁스를 vertex에서 보간
- **Vectorized bounding box**: 각 삼각형의 bounding box 내 모든 픽셀을 한 번에 계산

### 7.2. Crop ↔ Original 좌표 변환

Deep3DFaceRecon은 224×224 crop 공간에서 작동하므로,
결과를 원본 이미지 좌표로 변환해야 한다.

변환 방법:
1. `.mat`의 `lm68` (68-point landmarks, crop 공간) 에서 5-point 추출
2. `detections/*.txt`의 5-point landmark (원본 공간)과 매칭
3. `skimage.transform.SimilarityTransform`으로 변환 행렬 추정
4. `cv2.warpAffine`으로 노멀맵/뎁스맵/마스크를 원본 크기로 변환

### 7.3. 소프트 블렌딩

마스크 경계가 부자연스러운 것을 방지하기 위해:
1. `cv2.erode` (7×7 ellipse, 2 iterations): 마스크를 안쪽으로 축소
2. `cv2.GaussianBlur` (21×21, sigma=7): 경계를 부드럽게
3. 최종 합성: `result = original × (1 - mask) + silicone × mask`

---

## 8. 실행 방법 요약

```bash
cd /home/mmss9402/source/suprema/face_reconstruction

# 1. 랜드마크 검출 (이미 완료)
conda run -n deep3d python scripts/detect_landmarks.py

# 2. Deep3DFaceRecon inference (이미 완료)
cd methods/Deep3DFaceRecon_pytorch
conda run -n deep3d python test.py --name face_recon_feat0.2_augment --epoch 20 --img_folder ../../data/test_images
cd ../..

# 3. OBJ → PLY 변환 + MeshLab 확인 (이미 완료)
conda run -n deep3d python scripts/convert_and_compare.py

# 4. BFM 렌더링 정합 확인 (이미 완료)
conda run -n deep3d python scripts/render_deep3d.py

# 5. ★ 실리콘 재질 합성 (최종)
conda run -n deep3d python scripts/material_transfer.py
```

---

## 9. 현재 상태 및 다음 단계

### 완료
- [x] Deep3DFaceRecon, EMOCA 설치 및 inference
- [x] 3D 메시 MeshLab 확인
- [x] BFM 렌더링 정합 확인
- [x] 실리콘 재질 합성 (원본 이미지 보존 방식)

### 다음 단계 (TODO)
- [ ] 다른 재질 추가 (resin, latex, plaster)
- [ ] EMOCA 기반 동일 파이프라인 구현
- [ ] 대량 이미지 자동화 파이프라인
- [ ] PAD 모델 학습용 데이터셋 생성
