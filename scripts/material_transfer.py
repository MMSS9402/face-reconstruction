"""
Normal-based material transfer: 원본 이미지 보존 방식

3D 복원에서 기하 정보(노멀맵, 뎁스맵, 마스크)만 추출하고,
재질 효과는 원본 이미지 픽셀 위에 직접 적용한다.

사용법:
    conda run -n deep3d python scripts/material_transfer.py
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
from skimage.transform import SimilarityTransform

ROOT = Path(__file__).resolve().parent.parent
DEEP3D = ROOT / "methods" / "Deep3DFaceRecon_pytorch"
BFM_DIR = DEEP3D / "BFM"
MAT_DIR = (
    DEEP3D / "checkpoints" / "face_recon_feat0.2_augment"
    / "results" / "test_images" / "epoch_20_000000"
)
IMG_DIR = ROOT / "data" / "test_images"
OUT_DIR = ROOT / "outputs" / "deep3d_material"

FOCAL = 1015.0
CENTER = 112.0
CAMERA_D = 10.0
CROP_SIZE = 224


# ---------------------------------------------------------------------------
# BFM Model (geometry only -- texture computation is not needed here)
# ---------------------------------------------------------------------------

class BFMModel:
    def __init__(self, bfm_folder):
        model = loadmat(str(bfm_folder / "BFM_model_front.mat"))
        self.mean_shape = model["meanshape"].astype(np.float32)
        self.id_base = model["idBase"].astype(np.float32)
        self.exp_base = model["exBase"].astype(np.float32)
        self.face_buf = model["tri"].astype(np.int64) - 1
        self.point_buf = model["point_buf"].astype(np.int64) - 1
        self.keypoints = np.squeeze(model["keypoints"]).astype(np.int64) - 1

        ms = self.mean_shape.reshape(-1, 3)
        ms -= np.mean(ms, axis=0, keepdims=True)
        self.mean_shape = ms.reshape(-1, 1)

        self.persc_proj = np.array([
            FOCAL, 0, CENTER,
            0, FOCAL, CENTER,
            0, 0, 1,
        ], dtype=np.float32).reshape(3, 3).T

    def compute_shape(self, id_coeff, exp_coeff):
        shape = self.id_base @ id_coeff + self.exp_base @ exp_coeff + self.mean_shape.flatten()
        return shape.reshape(-1, 3)

    def compute_rotation(self, angles):
        x, y, z = angles
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return (Rz @ Ry @ Rx).T

    def compute_normals(self, vertices):
        f = self.face_buf
        v0, v1, v2 = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
        fn = np.cross(v0 - v1, v1 - v2)
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
        fn = np.vstack([fn, np.zeros((1, 3))])
        vn = np.sum(fn[self.point_buf], axis=1)
        return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)

    def reconstruct_geometry(self, mat_path):
        """Return 2D verts, rotated normals, camera-space depths, lm68."""
        m = loadmat(str(mat_path))
        id_c = m["id"].flatten()
        exp_c = m["exp"].flatten()
        angles = m["angle"].flatten()
        trans = m["trans"].flatten()
        lm68 = m["lm68"][0]

        shape = self.compute_shape(id_c, exp_c)
        rot = self.compute_rotation(angles)

        shape_t = shape @ rot + trans
        shape_cam = shape_t.copy()
        shape_cam[:, 2] = CAMERA_D - shape_cam[:, 2]

        proj = shape_cam @ self.persc_proj
        verts_2d = proj[:, :2] / proj[:, 2:3]
        verts_2d[:, 1] = CROP_SIZE - 1 - verts_2d[:, 1]

        normals_rot = self.compute_normals(shape) @ rot
        depths = shape_cam[:, 2]

        return verts_2d, normals_rot, depths, lm68


# ---------------------------------------------------------------------------
# Rasterizer: outputs normal map + depth map + mask in a single pass
# ---------------------------------------------------------------------------

def rasterize_geometry(verts_2d, normals, depths, faces, img_size=CROP_SIZE):
    """
    Single-pass rasterizer producing per-pixel normals, depth, and mask.
    Returns:
        normal_map (H,W,3), depth_map (H,W), mask (H,W)
    """
    normal_map = np.zeros((img_size, img_size, 3), dtype=np.float32)
    depth_map = np.full((img_size, img_size), np.inf, dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.float32)

    for f in faces:
        v0, v1, v2 = verts_2d[f[0]], verts_2d[f[1]], verts_2d[f[2]]
        n0, n1, n2 = normals[f[0]], normals[f[1]], normals[f[2]]
        z0, z1, z2 = depths[f[0]], depths[f[1]], depths[f[2]]

        xs = np.array([v0[0], v1[0], v2[0]])
        ys = np.array([v0[1], v1[1], v2[1]])
        x_min = max(int(np.floor(xs.min())), 0)
        x_max = min(int(np.ceil(xs.max())), img_size - 1)
        y_min = max(int(np.floor(ys.min())), 0)
        y_max = min(int(np.ceil(ys.max())), img_size - 1)
        if x_min > x_max or y_min > y_max:
            continue

        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-8:
            continue

        pxs = np.arange(x_min, x_max + 1)
        pys = np.arange(y_min, y_max + 1)
        gx, gy = np.meshgrid(pxs, pys)
        gx = gx.ravel().astype(np.float64)
        gy = gy.ravel().astype(np.float64)

        w0 = ((v1[1] - v2[1]) * (gx - v2[0]) + (v2[0] - v1[0]) * (gy - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (gx - v2[0]) + (v0[0] - v2[0]) * (gy - v2[1])) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        idx = np.where(inside)[0]
        px_in = gx[idx].astype(np.int32)
        py_in = gy[idx].astype(np.int32)
        w0i, w1i, w2i = w0[idx], w1[idx], w2[idx]
        zi = w0i * z0 + w1i * z1 + w2i * z2

        for i in range(len(idx)):
            px, py = px_in[i], py_in[i]
            if zi[i] < depth_map[py, px]:
                depth_map[py, px] = zi[i]
                normal_map[py, px] = w0i[i] * n0 + w1i[i] * n1 + w2i[i] * n2
                mask[py, px] = 1.0

    depth_map[mask == 0] = 0.0
    return normal_map, depth_map, mask


# ---------------------------------------------------------------------------
# Crop -> Original image coordinate transform
# ---------------------------------------------------------------------------

def compute_crop_transform(img_path, lm68_crop):
    det_path = img_path.parent / "detections" / img_path.with_suffix(".txt").name
    if not det_path.exists():
        return None

    lm5_orig = np.loadtxt(str(det_path))
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5_crop = np.stack([
        lm68_crop[lm_idx[0]],
        np.mean(lm68_crop[lm_idx[[1, 2]]], axis=0),
        np.mean(lm68_crop[lm_idx[[3, 4]]], axis=0),
        lm68_crop[lm_idx[5]],
        lm68_crop[lm_idx[6]],
    ])
    lm5_crop = lm5_crop[[1, 2, 0, 3, 4]]

    tform = SimilarityTransform()
    tform.estimate(lm5_crop, lm5_orig)
    return tform


# ---------------------------------------------------------------------------
# Material effect: Silicone (applied on ORIGINAL image)
# ---------------------------------------------------------------------------

def apply_silicone(orig_rgb, normal_map, depth_map, mask):
    """
    Apply realistic silicone mask effect directly on the original image.
    Based on real silicone mask visual characteristics:
    - Heavy surface smoothing (pores/wrinkles disappear)
    - Waxy, plastic-like surface with strong specular
    - Uniform skin tone (blood flow variation removed)
    - Slightly yellowish/pinkish "dead" color cast
    - Lifeless, doll-like appearance
    """
    result = orig_rgb.copy()
    m3 = mask[:, :, np.newaxis]

    # --- 1. Heavy surface smoothing: eliminate pores, wrinkles, fine detail ---
    # Real silicone masks have very smooth surfaces
    smooth1 = cv2.GaussianBlur(result, (15, 15), 5.0)
    smooth2 = cv2.bilateralFilter(
        (smooth1 * 255).astype(np.uint8), 9, 75, 75
    ).astype(np.float32) / 255.0
    face_smooth = result * 0.15 + smooth2 * 0.85
    result = result * (1 - m3) + face_smooth * m3

    # --- 2. Homogenize skin tone: remove local color variation ---
    # Real skin has micro color variation from blood flow; silicone doesn't
    local_mean = cv2.GaussianBlur(result, (31, 31), 12.0)
    deviation = result - local_mean
    face_uniform = local_mean + deviation * 0.3
    result = result * (1 - m3) + face_uniform * m3

    # --- 3. Significant desaturation: silicone looks washed-out ---
    gray = np.mean(result, axis=2, keepdims=True)
    face_desat = result * 0.65 + gray * 0.35
    result = result * (1 - m3) + face_desat * m3

    # --- 4. Waxy/yellowish color cast: "dead skin" look ---
    face_cast = result.copy()
    face_cast[:, :, 0] *= 1.06   # R up (pinkish/warm)
    face_cast[:, :, 1] *= 1.02   # G slightly up (yellowish)
    face_cast[:, :, 2] *= 0.88   # B significantly down (remove life)
    result = result * (1 - m3) + face_cast * m3

    # --- 5. Reduce contrast: silicone has flat, uniform albedo ---
    face_pixels = result[mask > 0.5]
    if len(face_pixels) > 0:
        mean_rgb = np.mean(face_pixels.reshape(-1, 3), axis=0)
    else:
        mean_rgb = np.array([0.5, 0.5, 0.5])
    face_flat = result * 0.7 + mean_rgb * 0.3
    result = result * (1 - m3) + face_flat * m3

    # --- 6. Strong plastic-like specular from 3D normals ---
    # Silicone has broad, prominent specular reflection (waxy sheen)
    half_vec = np.array([0.0, 0.0, 1.0])
    n_dot_h = np.clip(np.sum(normal_map * half_vec, axis=2), 0, 1)
    # Broad highlight (low exponent) + sharp highlight (high exponent)
    spec_broad = np.power(n_dot_h, 4) * 0.12
    spec_sharp = np.power(n_dot_h, 20) * 0.08
    specular = (spec_broad + spec_sharp) * mask
    result = result + specular[:, :, np.newaxis] * m3

    # --- 7. Slight overall brightness increase: silicone reflects more light ---
    face_bright = result * 1.05
    result = result * (1 - m3) + face_bright * m3

    return np.clip(result, 0, 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_one(bfm, img_name, out_dir):
    mat_path = MAT_DIR / f"{img_name}.mat"
    img_path = IMG_DIR / f"{img_name}.jpg"

    if not mat_path.exists() or not img_path.exists():
        print(f"  [SKIP] {img_name}: 파일 없음")
        return

    # --- Step 1: Reconstruct geometry from .mat ---
    verts_2d, normals_rot, depths, lm68 = bfm.reconstruct_geometry(mat_path)

    # --- Step 2: Rasterize normal/depth/mask in 224x224 crop space ---
    normal_crop, depth_crop, mask_crop = rasterize_geometry(
        verts_2d, normals_rot, depths, bfm.face_buf
    )

    # --- Step 3: Warp maps to original image coordinates ---
    tform = compute_crop_transform(img_path, lm68)
    if tform is None:
        print(f"  [SKIP] {img_name}: landmark 파일 없음")
        return

    orig_bgr = cv2.imread(str(img_path))
    h_orig, w_orig = orig_bgr.shape[:2]
    warp_size = (w_orig, h_orig)
    M = tform.params[:2]

    normal_orig = cv2.warpAffine(normal_crop, M, warp_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    depth_orig = cv2.warpAffine(depth_crop, M, warp_size,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
    mask_orig = cv2.warpAffine(mask_crop, M, warp_size,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    # Soft mask: erode + blur for smooth boundary blending
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_soft = cv2.erode(mask_orig, kernel, iterations=2)
    mask_soft = cv2.GaussianBlur(mask_soft, (21, 21), 7.0)

    # --- Step 3.5: Exclude eye/mouth holes (real masks have openings) ---
    lm68_orig = tform(lm68)  # (68,2) in original image coords

    # 68-landmark indices: left eye 36-41, right eye 42-47, mouth outer 48-59
    eye_l = lm68_orig[36:42].astype(np.int32)
    eye_r = lm68_orig[42:48].astype(np.int32)
    mouth = lm68_orig[48:60].astype(np.int32)

    for pts in [eye_l, eye_r]:
        center = pts.mean(axis=0)
        pts[:] = ((pts - center) * 1.4 + center).astype(np.int32)
    mouth_center = mouth.mean(axis=0)
    mouth = ((mouth - mouth_center) * 1.2 + mouth_center).astype(np.int32)

    hole_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
    cv2.fillPoly(hole_mask, [eye_l], 1.0)
    cv2.fillPoly(hole_mask, [eye_r], 1.0)
    cv2.fillPoly(hole_mask, [mouth], 1.0)
    hole_mask = cv2.GaussianBlur(hole_mask, (15, 15), 5.0)

    mask_soft = np.clip(mask_soft - hole_mask, 0, 1)

    # --- Step 4: Apply silicone material on original image ---
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    silicone_rgb = apply_silicone(orig_rgb, normal_orig, depth_orig, mask_soft)

    # --- Step 5: Save comparison ---
    silicone_bgr = cv2.cvtColor(
        (silicone_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    h_target = 512
    scale = h_target / h_orig

    def resize(im):
        return cv2.resize(im, (int(w_orig * scale), h_target))

    row = np.hstack([resize(orig_bgr), resize(silicone_bgr)])
    save_path = out_dir / f"{img_name}_silicone.png"
    cv2.imwrite(str(save_path), row)
    print(f"  [OK] {img_name} -> {save_path.name}  ({row.shape[1]}x{row.shape[0]})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("BFM 모델 로딩...")
    bfm = BFMModel(BFM_DIR)
    print(f"  vertices: {bfm.mean_shape.size // 3}, faces: {len(bfm.face_buf)}")

    mat_files = sorted(MAT_DIR.glob("*.mat"))
    print(f"\n{len(mat_files)}개 이미지 처리 중...")

    for mat_path in mat_files:
        process_one(bfm, mat_path.stem, OUT_DIR)

    print(f"\n완료! 결과: {OUT_DIR}")


if __name__ == "__main__":
    main()
