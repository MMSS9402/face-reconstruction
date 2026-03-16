# DECA Face Mask Composer

DECA 기반 3D 얼굴 복원을 사용해 첫 번째 이미지의 얼굴에서 `detail mesh overlay`를 만들고, 두 번째 이미지의 얼굴 landmark와 색 톤에 맞춰 자연스럽게 합성하는 웹앱입니다.

현재 메인 앱은 예전의 `silicone / latex / resin` 재질 합성 파이프라인이 아니라, 다음 흐름으로 동작합니다.

1. `source image`에서 DECA로 3D 얼굴 mesh를 복원합니다.
2. source crop 공간에서 `detail mesh overlay`를 RGBA로 렌더링합니다.
3. `target image`에서 얼굴 정렬 기준과 68 landmark를 추출합니다.
4. source landmark를 target landmark에 정렬한 뒤, target 얼굴 색 톤에 맞게 보정합니다.
5. 최종 `mask_composite.png`를 생성합니다.

## Current Pipeline

```text
Source Image
    │
    ▼
[DECA preprocessing]
    │
    ▼
[FLAME + Detail Decoder]
    │
    ├── detail.obj / detail.ply / coarse.ply / full_head.ply
    │
    ▼
[Detail Mesh Overlay Render in crop space]
    │
    ▼
mask_render.png

Target Image
    │
    ▼
[target face crop + 68 landmarks]
    │
    ▼
[source->target landmark alignment]
    │
    ▼
[target skin-tone matching in Lab space]
    │
    ▼
mask_composite.png
```

## Main Files

```text
scripts/webapp_deca.py            Flask 웹앱 메인 진입점
scripts/composite_blender.py      target landmark 정렬 + 색 톤 보정 + 합성
scripts/render_blender.py         보조 Blender 렌더 테스트 스크립트
scripts/test_blender_pipeline.py  단일 렌더/합성 테스트
README.md                         현재 앱 설명
```

## Requirements

- Ubuntu 20.04+
- Conda
- Python 3.9
- DECA + FLAME 데이터
- Blender 4.2 LTS (선택 사항: 보조 렌더 테스트용)

## Important Directory Layout

현재 코드(`scripts/webapp_deca.py`)는 아래 구조를 기대합니다.
이 구조대로 두면 추가 수정 없이 README만 따라 세팅할 수 있습니다.

```text
workspace/
├── synthetic_3d_attack/
│   └── third_party/
│       └── DECA/
│           ├── decalib/
│           └── data/
│               ├── deca_model.tar
│               ├── generic_model.pkl
│               ├── texture_data_256.npy
│               ├── fixed_displacement_256.npy
│               ├── uv_face_mask.png
│               └── uv_face_eye_mask.png
└── face_reconstruction/
    ├── scripts/
    ├── docs/
    └── README.md
```

즉 이 저장소(`face_reconstruction`)와 `synthetic_3d_attack`는 **같은 부모 폴더 아래 sibling**으로 있어야 합니다.

## Setup From Scratch

아래 순서를 처음부터 그대로 따라가면 됩니다.

### 1. 작업 폴더 준비

```bash
mkdir -p ~/workspace
cd ~/workspace
```

### 2. 이 저장소 배치

이 저장소를 `~/workspace/face_reconstruction` 에 둡니다.

예시:

```bash
cd ~/workspace
git clone <this-repo-url> face_reconstruction
```

이미 압축 해제나 복사본이 있다면, 최종 위치가 아래처럼 되면 됩니다.

```text
~/workspace/face_reconstruction
```

### 3. Conda 환경 생성

```bash
conda create -n deca-env python=3.9 -y
conda activate deca-env
```

### 4. PyTorch 설치

가장 이식성 있게 가려면 공식 가이드 대신 기본 설치로 시작하는 편이 안전합니다.

```bash
pip install torch torchvision
```

### 5. Python 의존성 설치

```bash
pip install flask==3.1.3 \
    opencv-python==4.13.0.92 \
    numpy==2.0.2 \
    scipy==1.13.1 \
    scikit-image==0.24.0 \
    trimesh==4.11.3 \
    face-alignment==1.4.1 \
    kornia==0.8.2 \
    matplotlib==3.9.4 \
    PyYAML==6.0.3 \
    yacs==0.1.8 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.10
```

### 6. DECA 설치

중요: DECA는 반드시 `face_reconstruction`와 같은 부모 폴더 아래의 `synthetic_3d_attack/third_party/DECA` 위치에 두어야 합니다.

```bash
cd ~/workspace
mkdir -p synthetic_3d_attack/third_party
cd synthetic_3d_attack/third_party
git clone https://github.com/yfeng95/DECA.git
cd DECA
bash fetch_data.sh
```

설치 후 최종 위치는 아래와 같아야 합니다.

```text
~/workspace/synthetic_3d_attack/third_party/DECA
```

### 7. Blender 설치

현재 메인 웹앱은 Blender 없이 동작합니다.
다만 `scripts/render_blender.py` 또는 `scripts/test_blender_pipeline.py`를 별도로 사용할 경우 Blender가 필요합니다.

```bash
cd ~/workspace
mkdir -p tools
cd tools
wget https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.13-linux-x64.tar.xz
tar xf blender-4.2.13-linux-x64.tar.xz
rm blender-4.2.13-linux-x64.tar.xz
```

## Preflight Check

서버를 띄우기 전에 아래 두 가지를 먼저 확인하세요.

### 1. 디렉토리 구조 확인

```bash
ls ~/workspace
```

적어도 아래 두 디렉토리가 보여야 합니다.

```text
face_reconstruction
synthetic_3d_attack
```

### 2. DECA 데이터 확인

```bash
ls ~/workspace/synthetic_3d_attack/third_party/DECA/data
```

아래 파일들이 있어야 합니다.

```text
deca_model.tar
generic_model.pkl
texture_data_256.npy
fixed_displacement_256.npy
uv_face_mask.png
uv_face_eye_mask.png
```

## Run

```bash
conda activate deca-env
cd ~/workspace/face_reconstruction
python scripts/webapp_deca.py
```

또는:

```bash
cd ~/workspace/face_reconstruction
conda run -n deca-env python scripts/webapp_deca.py
```

브라우저에서 `http://localhost:5000` 으로 접속합니다.

## Web App Inputs

- `source image`: 3D detail mesh overlay를 생성할 얼굴 이미지
- `target image`: 생성된 overlay를 합성할 원본 얼굴 이미지

권장 입력 조건:

- JPG 또는 PNG
- 얼굴이 충분히 크게 보이는 이미지
- 눈, 코, 입이 가려지지 않은 이미지
- 정면 또는 약한 측면 얼굴
- source와 target의 포즈 차이가 너무 크지 않은 이미지

## Outputs

웹앱 결과 페이지에서 다음 결과를 확인할 수 있습니다.

- `source_original.jpg`
- `target_original.jpg`
- `source_reconstruction.png`
- `mask_render.png`
- `mask_composite.png`
- `detail.obj`
- `detail.ply`
- `coarse.ply`
- `full_head.ply`

## Result Meaning

- `source_reconstruction.png`: source crop, detail mesh overlay, 68 landmarks를 나란히 보여주는 시각화
- `mask_render.png`: source crop 공간에서 렌더한 detail mesh overlay RGBA 결과
- `mask_composite.png`: target 얼굴 위치와 색 톤에 맞춰 보정 후 합성한 최종 결과

## Troubleshooting

### source 또는 target 얼굴 검출 실패

다음 경우 source 또는 target 이미지에서 얼굴을 찾지 못할 수 있습니다.

- 얼굴이 너무 작음
- 가림이 심함
- 측면 각도가 너무 큼
- 조명이 너무 어두움

### `.jpeg` 업로드 문제

DECA 내부 전처리는 확장자 처리에 민감합니다.
현재 웹앱은 업로드된 `.jpeg` 파일을 내부적으로 `.jpg`로 저장해 처리합니다.

### `No module named decalib` 또는 DECA 경로 오류

대부분 디렉토리 구조 문제입니다.
아래 위치를 다시 확인하세요.

```text
~/workspace/synthetic_3d_attack/third_party/DECA
~/workspace/face_reconstruction
```

두 디렉토리가 같은 부모(`~/workspace`) 아래 있어야 합니다.

### FLAME / DECA 데이터 경로 오류

DECA의 `data/` 아래에 `deca_model.tar`, `generic_model.pkl` 등이 있는지 확인하세요.

### 포트 5000 실행 실패

이미 다른 프로세스가 포트를 쓰고 있을 수 있습니다.
로컬 터미널에서 직접 실행해 확인하는 것이 가장 정확합니다.

## Legacy Notes

예전 `silicone / latex / resin / material transfer` 흐름과 Deep3DFaceRecon 기반 문서는 `docs/` 아래에 레거시 참고 자료로 남겨두었습니다. 현재 메인 앱 동작과는 다를 수 있습니다.
