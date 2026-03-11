"""
Deep3DFaceRecon .mat 계수로부터 3D 얼굴 복원 후
원본 시점에서 2D 렌더링 + 원본 이미지 overlay 확인

사용법:
    conda run -n deep3d python scripts/render_deep3d.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
DEEP3D = ROOT / "methods" / "Deep3DFaceRecon_pytorch"
BFM_DIR = DEEP3D / "BFM"
MAT_DIR = DEEP3D / "checkpoints" / "face_recon_feat0.2_augment" / "results" / "test_images" / "epoch_20_000000"
IMG_DIR = ROOT / "data" / "test_images"
OUT_DIR = ROOT / "outputs" / "deep3d_silicone"

FOCAL = 1015.0
CENTER = 112.0
CAMERA_D = 10.0
CROP_SIZE = 224


class BFMModel:
    """BFM parametric face model (numpy only, no torch)"""

    def __init__(self, bfm_folder):
        model = loadmat(str(bfm_folder / "BFM_model_front.mat"))
        self.mean_shape = model["meanshape"].astype(np.float32)     # (3N, 1)
        self.id_base = model["idBase"].astype(np.float32)           # (3N, 80)
        self.exp_base = model["exBase"].astype(np.float32)          # (3N, 64)
        self.mean_tex = model["meantex"].astype(np.float32)         # (3N, 1)
        self.tex_base = model["texBase"].astype(np.float32)         # (3N, 80)
        self.face_buf = model["tri"].astype(np.int64) - 1           # (F, 3)
        self.point_buf = model["point_buf"].astype(np.int64) - 1    # (N, 8)
        self.keypoints = np.squeeze(model["keypoints"]).astype(np.int64) - 1

        mean_shape = self.mean_shape.reshape(-1, 3)
        mean_shape -= np.mean(mean_shape, axis=0, keepdims=True)
        self.mean_shape = mean_shape.reshape(-1, 1)

        self.persc_proj = np.array([
            FOCAL, 0, CENTER,
            0, FOCAL, CENTER,
            0, 0, 1
        ], dtype=np.float32).reshape(3, 3).T

        # SH basis constants
        self.sh_a = [np.pi, 2 * np.pi / np.sqrt(3.0), 2 * np.pi / np.sqrt(8.0)]
        self.sh_c = [
            1 / np.sqrt(4 * np.pi),
            np.sqrt(3.0) / np.sqrt(4 * np.pi),
            3 * np.sqrt(5.0) / np.sqrt(12 * np.pi),
        ]
        self.init_lit = np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32).reshape(1, 1, 9)

    def compute_shape(self, id_coeff, exp_coeff):
        """(80,) + (64,) -> (N, 3) vertices"""
        id_part = self.id_base @ id_coeff     # (3N,)
        exp_part = self.exp_base @ exp_coeff   # (3N,)
        shape = id_part + exp_part + self.mean_shape.flatten()
        return shape.reshape(-1, 3)

    def compute_texture(self, tex_coeff):
        """(80,) -> (N, 3) rgb in [0,1]"""
        tex = self.tex_base @ tex_coeff + self.mean_tex.flatten()
        return (tex / 255.0).reshape(-1, 3)

    def compute_rotation(self, angles):
        """(3,) euler angles -> (3, 3) rotation"""
        x, y, z = angles
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return (Rz @ Ry @ Rx).T

    def compute_normals(self, vertices):
        """(N,3) -> (N,3) vertex normals"""
        f = self.face_buf
        v0, v1, v2 = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
        face_normals = np.cross(v0 - v1, v1 - v2)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / (norms + 1e-8)

        face_normals = np.vstack([face_normals, np.zeros((1, 3))])
        vertex_normals = np.sum(face_normals[self.point_buf], axis=1)
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        return vertex_normals / (norms + 1e-8)

    def compute_color(self, texture, normals, gamma):
        """Apply SH lighting to texture. gamma: (27,) -> (3,9)"""
        a, c = self.sh_a, self.sh_c
        gamma = gamma.reshape(3, 9) + self.init_lit.reshape(1, 9)
        n = normals
        Y = np.column_stack([
            a[0] * c[0] * np.ones(len(n)),
            -a[1] * c[1] * n[:, 1],
            a[1] * c[1] * n[:, 2],
            -a[1] * c[1] * n[:, 0],
            a[2] * c[2] * n[:, 0] * n[:, 1],
            -a[2] * c[2] * n[:, 1] * n[:, 2],
            0.5 * a[2] * c[2] / np.sqrt(3.0) * (3 * n[:, 2] ** 2 - 1),
            -a[2] * c[2] * n[:, 0] * n[:, 2],
            0.5 * a[2] * c[2] * (n[:, 0] ** 2 - n[:, 1] ** 2),
        ])  # (N, 9)
        r = Y @ gamma[0]
        g = Y @ gamma[1]
        b = Y @ gamma[2]
        lighting = np.column_stack([r, g, b])
        return np.clip(texture * lighting, 0, 1)

    def reconstruct(self, mat_path):
        """Load .mat and reconstruct everything"""
        m = loadmat(str(mat_path))
        id_c = m["id"].flatten()
        exp_c = m["exp"].flatten()
        tex_c = m["tex"].flatten()
        angles = m["angle"].flatten()
        gamma = m["gamma"].flatten()
        trans = m["trans"].flatten()

        shape = self.compute_shape(id_c, exp_c)
        rot = self.compute_rotation(angles)
        shape_transformed = shape @ rot + trans
        shape_camera = shape_transformed.copy()
        shape_camera[:, 2] = CAMERA_D - shape_camera[:, 2]

        proj = shape_camera @ self.persc_proj
        verts_2d = proj[:, :2] / proj[:, 2:3]
        verts_2d[:, 1] = CROP_SIZE - 1 - verts_2d[:, 1]

        texture = self.compute_texture(tex_c)
        normals = self.compute_normals(shape)
        normals_rot = normals @ rot
        color = self.compute_color(texture, normals_rot, gamma)

        return verts_2d, color, shape_camera


def rasterize(verts_2d, colors, faces, depths, img_size=CROP_SIZE):
    """
    Numpy triangle rasterizer with proper z-buffer and backface culling.
    depths: per-vertex camera-space z (smaller = closer to camera)
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    zbuf = np.full((img_size, img_size), np.inf, dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.float32)

    for f in faces:
        v0, v1, v2 = verts_2d[f[0]], verts_2d[f[1]], verts_2d[f[2]]
        c0, c1, c2 = colors[f[0]], colors[f[1]], colors[f[2]]
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

        # vectorized: generate all pixel coords in bounding box
        pys = np.arange(y_min, y_max + 1)
        pxs = np.arange(x_min, x_max + 1)
        grid_px, grid_py = np.meshgrid(pxs, pys)
        grid_px = grid_px.ravel().astype(np.float64)
        grid_py = grid_py.ravel().astype(np.float64)

        w0 = ((v1[1] - v2[1]) * (grid_px - v2[0]) + (v2[0] - v1[0]) * (grid_py - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (grid_px - v2[0]) + (v0[0] - v2[0]) * (grid_py - v2[1])) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        idx = np.where(inside)[0]
        px_in = grid_px[idx].astype(np.int32)
        py_in = grid_py[idx].astype(np.int32)
        w0_in, w1_in, w2_in = w0[idx], w1[idx], w2[idx]

        z_interp = w0_in * z0 + w1_in * z1 + w2_in * z2

        for i in range(len(idx)):
            px, py = px_in[i], py_in[i]
            z = z_interp[i]
            if z < zbuf[py, px]:
                zbuf[py, px] = z
                img[py, px] = w0_in[i] * c0 + w1_in[i] * c1 + w2_in[i] * c2
                mask[py, px] = 1.0

    return img, mask


def apply_silicone_material(render_rgb, normals_img, mask, orig_crop_rgb):
    """
    Apply silicone mask material effect to the rendered face.

    render_rgb:    (H, W, 3) float [0,1] - BFM rendered face
    normals_img:   (H, W, 3) float - per-pixel surface normals (camera space)
    mask:          (H, W) float [0,1] - face mask
    orig_crop_rgb: (H, W, 3) float [0,1] - original cropped image (for detail reference)
    """
    h, w = render_rgb.shape[:2]
    result = render_rgb.copy()
    m = mask[:, :, np.newaxis]

    # 1. Surface smoothing: subtle pore/wrinkle reduction
    smooth = cv2.GaussianBlur(result, (7, 7), 2.0)
    result = result * 0.5 + smooth * 0.5

    # 2. Slight desaturation: silicone has slightly less color variation
    gray = np.mean(result, axis=2, keepdims=True)
    result = result * 0.85 + gray * 0.15

    # 3. Color shift: very subtle warm tone
    result[:, :, 0] *= 1.01  # R barely up
    result[:, :, 1] *= 0.99  # G barely down
    result[:, :, 2] *= 0.97  # B slightly down

    # 4. Reduce contrast: slightly more uniform albedo
    mean_val = np.mean(result[mask > 0.5]) if np.any(mask > 0.5) else 0.5
    result = result * 0.85 + mean_val * 0.15

    # 5. Specular highlights: subtle broad specular
    if normals_img is not None:
        half_vec = np.array([0.0, 0.0, 1.0])
        n_dot_h = np.clip(np.sum(normals_img * half_vec, axis=2), 0, 1)
        specular = np.power(n_dot_h, 10) * 0.08
        result = result + specular[:, :, np.newaxis]

    # 6. Subtle subsurface scattering simulation
    sss = cv2.GaussianBlur(result, (15, 15), 5.0)
    sss[:, :, 0] *= 1.05
    result = result * 0.92 + sss * 0.08

    return np.clip(result, 0, 1)


def rasterize_normals(verts_2d, normals, faces, depths, img_size=CROP_SIZE):
    """Rasterize per-pixel normals (same z-buffer logic as rasterize)"""
    nimg = np.zeros((img_size, img_size, 3), dtype=np.float32)
    zbuf = np.full((img_size, img_size), np.inf, dtype=np.float32)

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

        pys = np.arange(y_min, y_max + 1)
        pxs = np.arange(x_min, x_max + 1)
        gx, gy = np.meshgrid(pxs, pys)
        gx, gy = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)

        w0 = ((v1[1]-v2[1])*(gx-v2[0]) + (v2[0]-v1[0])*(gy-v2[1])) / denom
        w1 = ((v2[1]-v0[1])*(gx-v2[0]) + (v0[0]-v2[0])*(gy-v2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        idx = np.where(inside)[0]
        px_in, py_in = gx[idx].astype(int), gy[idx].astype(int)
        w0i, w1i, w2i = w0[idx], w1[idx], w2[idx]
        zi = w0i * z0 + w1i * z1 + w2i * z2

        for i in range(len(idx)):
            px, py = px_in[i], py_in[i]
            if zi[i] < zbuf[py, px]:
                zbuf[py, px] = zi[i]
                nimg[py, px] = w0i[i] * n0 + w1i[i] * n1 + w2i[i] * n2

    return nimg


def compute_crop_transform(img_path, lm68_crop):
    """
    Compute the affine transform from 224x224 crop back to original image.
    Uses the 5-point landmarks from detection files + the lm68 from .mat
    """
    det_path = img_path.parent / "detections" / img_path.with_suffix(".txt").name
    if not det_path.exists():
        return None, None

    lm5_orig = np.loadtxt(str(det_path))  # (5, 2) in original image

    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5_crop = np.stack([
        lm68_crop[lm_idx[0]],
        np.mean(lm68_crop[lm_idx[[1, 2]]], axis=0),
        np.mean(lm68_crop[lm_idx[[3, 4]]], axis=0),
        lm68_crop[lm_idx[5]],
        lm68_crop[lm_idx[6]],
    ])
    lm5_crop = lm5_crop[[1, 2, 0, 3, 4]]

    from skimage.transform import SimilarityTransform
    tform = SimilarityTransform()
    tform.estimate(lm5_crop, lm5_orig)
    return tform, lm5_orig


def process_one(bfm, img_name, out_dir):
    mat_path = MAT_DIR / f"{img_name}.mat"
    img_path = IMG_DIR / f"{img_name}.jpg"

    if not mat_path.exists() or not img_path.exists():
        print(f"  [SKIP] {img_name}: 파일 없음")
        return

    m = loadmat(str(mat_path))
    lm68 = m["lm68"][0]  # (68, 2)

    verts_2d, colors, shape_cam = bfm.reconstruct(mat_path)
    depths = shape_cam[:, 2]

    # Reconstruct normals for specular calculation
    m2 = loadmat(str(mat_path))
    angles = m2["angle"].flatten()
    rot = bfm.compute_rotation(angles)
    id_c, exp_c = m2["id"].flatten(), m2["exp"].flatten()
    shape = bfm.compute_shape(id_c, exp_c)
    normals_rot = bfm.compute_normals(shape) @ rot

    render, mask = rasterize(verts_2d, colors, bfm.face_buf, depths)
    normals_img = rasterize_normals(verts_2d, normals_rot, bfm.face_buf, depths)

    # Apply silicone material
    silicone = apply_silicone_material(render, normals_img, mask, render)

    render_uint8 = (np.clip(render, 0, 1) * 255).astype(np.uint8)
    silicone_uint8 = (np.clip(silicone, 0, 1) * 255).astype(np.uint8)
    render_bgr = cv2.cvtColor(render_uint8, cv2.COLOR_RGB2BGR)
    silicone_bgr = cv2.cvtColor(silicone_uint8, cv2.COLOR_RGB2BGR)

    orig_img = cv2.imread(str(img_path))
    h_orig, w_orig = orig_img.shape[:2]

    tform, _ = compute_crop_transform(img_path, lm68)
    if tform is None:
        print(f"  [SKIP] {img_name}: landmark 파일 없음")
        return

    silicone_warped = cv2.warpAffine(
        silicone_bgr, tform.params[:2],
        (w_orig, h_orig),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask_warped = cv2.warpAffine(
        mask, tform.params[:2],
        (w_orig, h_orig),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Soft edge blending: erode then blur the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(mask_warped, kernel, iterations=2)
    mask_soft = cv2.GaussianBlur(mask_eroded, (15, 15), 5.0)
    mask_3ch = mask_soft[:, :, np.newaxis]

    silicone_composite = (orig_img * (1 - mask_3ch) + silicone_warped * mask_3ch).astype(np.uint8)

    h_target = 512
    scale = h_target / h_orig

    def resize(im):
        return cv2.resize(im, (int(w_orig * scale), h_target))

    row = np.hstack([resize(orig_img), resize(silicone_composite)])

    save_path = out_dir / f"{img_name}_overlay.png"
    cv2.imwrite(str(save_path), row)
    print(f"  [OK] {img_name} -> {save_path.name}  ({row.shape[1]}x{row.shape[0]})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("BFM 모델 로딩...")
    bfm = BFMModel(BFM_DIR)
    print(f"  vertices: {bfm.mean_shape.size // 3}, faces: {len(bfm.face_buf)}")

    mat_files = sorted(MAT_DIR.glob("*.mat"))
    print(f"\n{len(mat_files)}개 이미지 렌더링 중...")

    for mat_path in mat_files:
        name = mat_path.stem
        process_one(bfm, name, OUT_DIR)

    print(f"\n완료! 결과: {OUT_DIR}")


if __name__ == "__main__":
    main()
