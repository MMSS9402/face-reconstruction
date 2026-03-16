# Current App And Legacy Pipeline Notes

> 최종 업데이트: 2026-03-16

이 문서는 현재 메인 웹앱의 동작과, 저장소 안에 남아 있는 예전 Deep3DFaceRecon / material transfer 파이프라인의 관계를 정리합니다.

## 1. 현재 메인 앱

현재 실제 메인 앱은 `scripts/webapp_deca.py` 입니다.

핵심 특징:
- 입력은 2장입니다: `source image`, `target image`
- `source`에서 DECA 기반 3D 얼굴 복원 수행
- `detail mesh overlay`를 source crop 공간에서 렌더링
- `target`의 68 landmark 기준으로 정렬 후 합성
- `target` 얼굴 톤에 맞춘 색 보정 수행
- 결과로 `mask_composite.png` 생성

### 현재 메인 플로우

```text
source image
  -> DECA inference
  -> detail mesh / source_reconstruction.png / mask_render.png

target image
  -> target face crop + 68 landmarks
  -> source render alignment
  -> tone matching
  -> mask_composite.png
```

### 현재 주요 산출물

- `source_original.jpg`
- `target_original.jpg`
- `source_reconstruction.png`
- `mask_render.png`
- `mask_composite.png`
- `detail.obj`
- `detail.ply`
- `coarse.ply`
- `full_head.ply`

### 현재 주요 파일

- `scripts/webapp_deca.py`
- `scripts/composite_blender.py`
- `README.md`

## 2. 예전 Deep3D / Material Transfer 경로

저장소에는 아래와 같은 예전 실험용 또는 레거시 경로가 남아 있습니다.

- `scripts/detect_landmarks.py`
- `scripts/render_deep3d.py`
- `scripts/material_transfer.py`
- `methods/Deep3DFaceRecon_pytorch/`
- `scripts/webapp.py`

이 경로들은 주로 다음 목적을 가졌습니다.

- Deep3DFaceRecon 기반 3D 복원
- 실리콘 재질 합성
- 재질 실험용 후처리
- Deep3D / EMOCA 비교

현재 메인 웹앱은 이 경로를 직접 사용하지 않습니다.

## 3. 레거시 문서를 이렇게 봐야 합니다

### `render_deep3d.py`
- Deep3DFaceRecon 계수 기반 렌더링 정합 확인용
- 현재 메인 앱의 source-target 합성 로직과는 다름

### `material_transfer.py`
- 실리콘 재질 시각 효과를 원본 이미지에 직접 적용하는 예전 방식
- 현재 메인 앱의 `detail mesh overlay` 합성과는 목적이 다름

### `scripts/webapp.py`
- 예전 웹앱 엔트리
- 현재는 `scripts/webapp_deca.py`가 메인

## 4. 현재 앱과 레거시 파이프라인 차이

| 항목 | 현재 메인 앱 | 예전 Deep3D / Material Transfer |
|------|--------------|----------------------------------|
| 메인 엔트리 | `scripts/webapp_deca.py` | `scripts/webapp.py`, `material_transfer.py` 등 |
| 입력 | source 1장 + target 1장 | 보통 단일 입력 이미지 |
| 3D 복원 | DECA | Deep3DFaceRecon 또는 EMOCA 비교 |
| 최종 결과 | `detail mesh overlay`의 target 합성 | 실리콘/재질 효과 합성 |
| 색 조정 | target 피부 톤 매칭 | 재질 기반 시각 효과 |
| 목적 | source 얼굴 형상을 target에 자연스럽게 옮기기 | 마스크 재질 시뮬레이션 |

## 5. 현재 문서를 볼 때 우선순위

현재 앱을 이해하거나 수정할 때는 아래 순서로 보는 것이 맞습니다.

1. `README.md`
2. `scripts/webapp_deca.py`
3. `scripts/composite_blender.py`
4. 필요 시 `scripts/render_blender.py`, `scripts/test_blender_pipeline.py`
5. 그 다음에만 레거시 Deep3D 문서 참고

## 6. 정리

현재 저장소의 중심은 더 이상 `silicone / resin / latex` 재질 합성이 아닙니다.
지금 메인 앱은 `source detail mesh overlay -> target landmark alignment -> tone matched composite` 구조로 바뀌었습니다.
