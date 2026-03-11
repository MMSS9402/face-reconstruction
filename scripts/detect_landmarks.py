"""
MTCNN으로 이미지에서 5-point facial landmark를 검출하여
Deep3DFaceRecon_pytorch 입력 형식(detections/*.txt)으로 저장.

사용법:
    conda run -n deep3d python detect_landmarks.py --img_dir ../data/test_images
"""

import argparse
from pathlib import Path

import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image


def detect_landmarks(img_dir: Path):
    det_dir = img_dir / "detections"
    det_dir.mkdir(exist_ok=True)

    mtcnn = MTCNN(keep_all=False, device="cpu")

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    print(f"이미지 {len(images)}장에서 5-point landmark 검출 중...")

    success = 0
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if landmarks is None or len(landmarks) == 0:
            print(f"  [SKIP] {img_path.name}: 얼굴 미검출")
            continue

        lm = landmarks[0]  # shape (5, 2): left_eye, right_eye, nose, left_mouth, right_mouth
        txt_path = det_dir / img_path.with_suffix(".txt").name
        np.savetxt(str(txt_path), lm, fmt="%.6f")
        success += 1
        print(f"  [OK] {img_path.name} → {txt_path.name}")

    print(f"\n완료: {success}/{len(images)} 검출 성공")
    print(f"결과: {det_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()
    detect_landmarks(Path(args.img_dir))
