"""
Composite Blender RGBA renders onto original images.

Takes a Blender rendered face mask (224x224 RGBA in source crop space),
aligns it to the target crop with landmarks, and composites it onto the
target image using the target DECA transform.

Usage (standalone test):
    python composite_blender.py \
        --render_dir /path/to/blender_renders/ \
        --original /path/to/target.jpg \
        --target_tform_npy /path/to/target_tform_params.npy \
        --target_landmarks_npy /path/to/target_lm2d.npy \
        --output_dir /path/to/output/
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from skimage.transform import SimilarityTransform


def landmarks_to_crop_pixels(lm2d, crop_size):
    return (lm2d + 1.0) / 2.0 * crop_size


def align_render_to_target_crop(render_rgba, source_lm2d, target_lm2d, crop_size):
    """Align the source crop render to the target crop using landmarks."""
    if source_lm2d is None or target_lm2d is None:
        return render_rgba
    if len(source_lm2d) != len(target_lm2d):
        return render_rgba

    src_pts = landmarks_to_crop_pixels(source_lm2d, crop_size).astype(np.float64)
    dst_pts = landmarks_to_crop_pixels(target_lm2d, crop_size).astype(np.float64)

    tform = SimilarityTransform()
    if not tform.estimate(src_pts, dst_pts):
        return render_rgba

    M = tform.params[:2].astype(np.float32)
    return cv2.warpAffine(
        render_rgba,
        M,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def composite_render(
    original_bgr,
    render_rgba,
    tform_params,
    target_lm2d,
    source_lm2d=None,
    crop_size=224,
):
    """Composite a 224x224 RGBA render onto the target image."""
    h, w = original_bgr.shape[:2]

    render_aligned = align_render_to_target_crop(
        render_rgba, source_lm2d, target_lm2d, crop_size
    )

    rgb = render_aligned[:, :, :3].astype(np.float32)
    alpha = render_aligned[:, :, 3].astype(np.float32) / 255.0

    tform = SimilarityTransform()
    tform.params = tform_params.astype(np.float64)
    M = tform.params[:2]

    warp_flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    rgb_orig = cv2.warpAffine(rgb, M, (w, h),
                               flags=warp_flags,
                               borderMode=cv2.BORDER_CONSTANT)
    alpha_orig = cv2.warpAffine(alpha, M, (w, h),
                                 flags=warp_flags,
                                 borderMode=cv2.BORDER_CONSTANT)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha_orig = cv2.erode(alpha_orig, kernel, iterations=1)
    alpha_orig = cv2.GaussianBlur(alpha_orig, (15, 15), 5.0)

    lm_crop_px = landmarks_to_crop_pixels(target_lm2d, crop_size)
    lm_orig = tform.inverse(lm_crop_px).astype(np.int32)

    eye_l = lm_orig[36:42].copy()
    eye_r = lm_orig[42:48].copy()
    mouth = lm_orig[48:60].copy()

    for pts in [eye_l, eye_r]:
        c = pts.mean(axis=0)
        pts[:] = ((pts - c) * 1.5 + c).astype(np.int32)
    mc = mouth.mean(axis=0)
    mouth = ((mouth - mc) * 1.3 + mc).astype(np.int32)

    hole = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(hole, [eye_l], 1.0)
    cv2.fillPoly(hole, [eye_r], 1.0)
    cv2.fillPoly(hole, [mouth], 1.0)
    hole = cv2.GaussianBlur(hole, (11, 11), 4.0)
    alpha_orig = np.clip(alpha_orig - hole, 0, 1)

    a3 = alpha_orig[:, :, np.newaxis]
    result = original_bgr.astype(np.float32) * (1 - a3) + rgb_orig * a3

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--target_tform_npy", required=True)
    parser.add_argument("--target_landmarks_npy", required=True)
    parser.add_argument("--source_landmarks_npy")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = cv2.imread(args.original)
    tform_params = np.load(args.target_tform_npy)
    target_lm2d = np.load(args.target_landmarks_npy)
    source_lm2d = None
    if args.source_landmarks_npy:
        source_lm2d = np.load(args.source_landmarks_npy)

    render_path = render_dir / "mask_render.png"
    if not render_path.exists():
        print(f"  [SKIP] {render_path} not found")
        return

    render_rgba = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
    result = composite_render(
        original_bgr,
        render_rgba,
        tform_params,
        target_lm2d,
        source_lm2d=source_lm2d,
    )
    out_path = output_dir / "mask_composite.png"
    cv2.imwrite(str(out_path), result)
    print(f"  [OK] {out_path}")


if __name__ == "__main__":
    main()
