# Face Reconstruction App Current Plan Snapshot

> 최종 업데이트: 2026-03-16

이 문서는 현재 애플리케이션의 구현 방향을 짧게 정리한 스냅샷입니다.
이전의 재질 합성 중심 계획이 아니라, 현재 반영된 `2-image detail mesh overlay compositing` 기준으로 작성되었습니다.

## 목표

두 장의 이미지를 입력으로 받아:
- `source image`에서 3D 얼굴 detail mesh를 복원하고
- 그 `detail mesh overlay`를
- `target image`의 얼굴 landmark와 피부 톤에 맞춰
- 자연스럽게 합성하는 것

## 현재 구현된 구조

### 입력
- `source image`
- `target image`

### source 처리
- DECA 전처리
- FLAME + detail decoder로 3D 복원
- `detail.obj`, `detail.ply`, `coarse.ply`, `full_head.ply` 생성
- `source_reconstruction.png` 생성
- crop 공간에서 `mask_render.png` 생성

### target 처리
- target 얼굴 crop 및 68 landmark 추출
- source landmark를 target landmark에 정렬
- target 피부 톤 기준으로 render 색 보정
- `mask_composite.png` 생성

## 현재 주요 파일

- `scripts/webapp_deca.py`
- `scripts/composite_blender.py`
- `scripts/render_blender.py`
- `scripts/test_blender_pipeline.py`
- `README.md`

## 결과물

- `source_original.jpg`
- `target_original.jpg`
- `source_reconstruction.png`
- `mask_render.png`
- `mask_composite.png`
- `detail.obj`
- `detail.ply`
- `coarse.ply`
- `full_head.ply`

## 구현 포인트

### 정렬
- source 얼굴의 68 landmark를 target의 68 landmark에 similarity transform으로 정렬
- 정렬된 render를 target 이미지 좌표계로 warp

### 색 자연화
- target crop의 얼굴 영역 통계를 기준으로 Lab 색공간 톤 매칭
- 눈/입 영역은 제외하여 피부 톤 위주로 보정

### 유지한 기능
- 3D mesh 뷰어
- OBJ / PLY 다운로드
- source 복원 결과 시각화

## 남은 개선 여지

- source와 target 포즈 차이가 큰 경우 실루엣 부자연스러움 감소
- 가장자리 feather / alpha 보정
- 색 보정 강도 사용자 조절
- landmark 기반 similarity transform보다 더 정교한 비강체 보정
