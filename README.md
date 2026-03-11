# 3D Face Reconstruction & Mask Material Synthesis

DECA 기반 3D 얼굴 복원 + Blender PBR 렌더링으로 실리콘/라텍스/레진 마스크 재질을 합성하는 파이프라인.

iBeta Level 2 인증을 위한 3D 마스크 공격 탐지 학습용 합성 데이터를 생성합니다.

## Pipeline

```
Input Image
    │
    ▼
[FAN Landmark Detection] ─── 68-point 랜드마크
    │
    ▼
[DECA Encoder] ─── FLAME shape/expression/pose 파라미터 + detail code
    │
    ▼
[FLAME Decode] ─── Coarse mesh (5K vertices)
    │
    ▼
[Detail Decoder] ─── UV displacement map → Detail mesh (~21K face-only vertices)
    │
    ▼
[Vertex Color Sampling] ─── Orthographic projection으로 원본에서 텍스처 추출
    │
    ▼
[Blender Cycles PBR] ─── 실리콘/라텍스/레진 3종 재질 렌더링 (224×224, DECA crop space)
    │                      Principled BSDF + SSS + 스튜디오 조명
    ▼
[Composite] ─── Blender 렌더를 원본 이미지에 합성 (affine warp + alpha blending)
    │
    ▼
Output: 원본 | 실리콘 합성 | 라텍스 합성 | 레진 합성
```

## Requirements

- **OS**: Ubuntu 20.04+ (Linux)
- **GPU**: NVIDIA GPU (CUDA 지원, RTX 3090 권장)
- **NVIDIA Driver**: 550+
- **Conda**: Anaconda 또는 Miniconda
- **Blender**: 4.2 LTS (자동 다운로드 스크립트 제공)
- **Disk**: ~10GB (DECA 모델 + Blender + FLAME 데이터)

## Project Structure

```
face_reconstruction/
├── scripts/
│   ├── webapp_deca.py          # Flask 웹앱 (메인 진입점)
│   ├── render_blender.py       # Blender PBR 재질 렌더링 스크립트
│   ├── composite_blender.py    # Blender 렌더 → 원본 합성
│   └── test_blender_pipeline.py # 파이프라인 테스트 스크립트
├── data/
│   └── test_images/            # 테스트용 샘플 이미지
├── docs/                       # 문서
└── README.md
```

## Setup

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/face-reconstruction.git
cd face-reconstruction
```

### 2. Conda 환경 생성

```bash
conda create -n deca-env python=3.9 -y
conda activate deca-env
```

### 3. PyTorch 설치 (CUDA 12.1)

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

> **Note**: GPU에 맞는 CUDA 버전을 선택하세요. [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/) 참고.

### 4. Python 의존성 설치

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

### 5. DECA 설치

DECA는 별도의 서드파티 모듈입니다.

```bash
# 프로젝트 루트 기준
mkdir -p third_party && cd third_party
git clone https://github.com/yfeng95/DECA.git
cd DECA
```

#### FLAME 모델 데이터 다운로드

1. [FLAME 공식 사이트](https://flame.is.tue.mpg.de/)에 가입
2. DECA의 `fetch_data.sh` 실행:

```bash
bash fetch_data.sh
```

이 스크립트가 다음을 다운로드합니다:
- `data/generic_model.pkl` — FLAME 모델
- `data/deca_model.tar` — DECA 사전 학습 가중치

다운로드 후 `data/` 디렉토리 구조:
```
DECA/data/
├── generic_model.pkl
├── deca_model.tar
├── FLAME2020/
├── fixed_displacement_256.npy
├── head_template.obj
├── landmark_embedding.npy
├── mean_texture.jpg
├── texture_data_256.npy
├── uv_face_eye_mask.png
├── uv_face_mask.png
└── extra_models/
```

### 6. Blender 설치 (Headless)

```bash
# 프로젝트 루트로 이동
cd /path/to/face-reconstruction

# Blender 4.2 LTS 다운로드 및 설치
mkdir -p tools && cd tools
wget https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.13-linux-x64.tar.xz
tar xf blender-4.2.13-linux-x64.tar.xz
rm blender-4.2.13-linux-x64.tar.xz
cd ..
```

설치 확인:
```bash
./tools/blender-4.2.13-linux-x64/blender --version
# → Blender 4.2.13 LTS
```

### 7. 경로 설정 확인

`scripts/webapp_deca.py`에서 다음 경로들이 올바른지 확인하세요:

```python
ROOT = Path(__file__).resolve().parent.parent          # face_reconstruction/
DECA_DIR = ROOT.parent / "third_party" / "DECA"        # DECA 위치
BLENDER_PATH = ROOT.parent / "tools" / "blender-4.2.13-linux-x64" / "blender"
```

프로젝트 디렉토리 구조가 다음과 같아야 합니다:
```
project-root/
├── face_reconstruction/    # 이 저장소
│   └── scripts/
├── third_party/
│   └── DECA/              # DECA (fetch_data.sh 실행 완료)
└── tools/
    └── blender-4.2.13-linux-x64/
```

## Run (실행)

### 웹앱 서버 시작

```bash
conda activate deca-env
cd face_reconstruction
conda run -n deca-env python scripts/webapp_deca.py
```

서버가 시작되면 다음과 같이 출력됩니다:
```
============================================================
  DECA Face Reconstruction 초기화 중...
============================================================

[1/1] DECA 모델 로딩...
  DECA 모델 로딩 중...
  Face-only filter: 21,440/39,470 vertices (41,986/77,400 faces)
  DECA 로딩 완료!

============================================================
  서버 시작: http://localhost:5000
============================================================
```

### 웹앱 접속

브라우저에서 **http://localhost:5000** 으로 접속합니다.

1. 얼굴 사진을 업로드합니다 (JPG/PNG)
2. 약 15~20초 후 결과가 표시됩니다:
   - **3D Detail Mesh 뷰어** — Three.js 인터랙티브 3D 뷰어
   - **3D 복원 결과** — 원본 크롭 / 메시 오버레이 / 랜드마크
   - **마스크 재질 합성** — 원본 / 실리콘 / 라텍스 / 레진
3. 메시(OBJ/PLY) 및 합성 이미지를 다운로드할 수 있습니다

### 백그라운드 실행

서버를 백그라운드에서 실행하려면:

```bash
# 백그라운드 실행 (로그를 파일로 저장)
nohup conda run -n deca-env python scripts/webapp_deca.py > server.log 2>&1 &

# 프로세스 ID 확인
echo $!
```

## Stop (종료)

### 포그라운드 실행 중일 때

터미널에서 `Ctrl+C`를 누르면 서버가 종료됩니다.

### 백그라운드 실행 중일 때

```bash
# 방법 1: PID로 종료 (서버 시작 시 출력된 PID 사용)
kill <PID>

# 방법 2: 포트 5000번을 사용하는 프로세스 찾아서 종료
lsof -ti :5000 | xargs kill

# 방법 3: 프로세스 이름으로 찾아서 종료
ps aux | grep webapp_deca | grep -v grep
kill <찾은_PID>
```

### 서버가 완전히 종료됐는지 확인

```bash
# 포트 5000번 사용 중인 프로세스가 없으면 종료 완료
lsof -i :5000
# (출력 없으면 종료됨)
```

## Material Presets

| 재질 | SSS | Roughness | Specular | Coat | 특징 |
|------|-----|-----------|----------|------|------|
| **Silicone** | 0.55 | 0.38 | 0.65 | 0.20 | 높은 반투명, 피부 아래 빛 번짐, 부드러운 반광택 |
| **Latex** | 0.12 | 0.60 | 0.35 | 0.08 | 매트한 고무 질감, 약한 광택 |
| **Resin** | 0.03 | 0.08 | 0.85 | 0.55 | 강한 플라스틱 광택, 하드한 반사, SSS 거의 없음 |

## Troubleshooting

### CUDA out of memory
GPU 메모리가 부족할 경우 `webapp_deca.py`의 디바이스를 CPU로 변경:
```python
deca_model = DECAModel(device="cpu")
```

### Blender GPU 렌더링 실패
Blender가 CUDA GPU를 인식하지 못하면 `render_blender.py`의 `setup_render()`에서:
```python
prefs.compute_device_type = "CUDA"  # → "OPTIX" 또는 "NONE"으로 변경
```

### FLAME 모델 경로 오류
`deca_model.tar`와 `generic_model.pkl`이 DECA의 `data/` 디렉토리에 있는지 확인하세요.

### 포트 충돌
5000번 포트가 이미 사용 중이면:
```bash
# 사용 중인 프로세스 확인
lsof -i :5000

# 다른 포트로 실행
conda run -n deca-env python scripts/webapp_deca.py  # webapp_deca.py 내 port=5000을 변경
```

## License

이 프로젝트는 연구 목적으로 작성되었습니다.
- [DECA](https://github.com/yfeng95/DECA) — FLAME 라이선스 준수 필요
- [FLAME](https://flame.is.tue.mpg.de/) — 비상업적 연구 라이선스
- [Blender](https://www.blender.org/) — GPL v2
