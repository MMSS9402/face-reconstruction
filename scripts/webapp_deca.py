"""
Flask 웹앱: DECA 3D Face Reconstruction + Detail Mesh
renderer 없이 DECA 인코더 + FLAME + Detail Decoder만 사용

사용법:
    conda run -n deca-env python scripts/webapp_deca.py
    → http://localhost:5000
"""

import sys
import os
import uuid
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from flask import Flask, request, render_template_string, send_file

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DECA_DIR = ROOT.parent / "synthetic_3d_attack" / "third_party" / "DECA"
UPLOAD_DIR = ROOT / "webapp_uploads"
RESULT_DIR = ROOT / "webapp_results"

sys.path.insert(0, str(DECA_DIR))
sys.path.insert(0, str(ROOT / "scripts"))


# ===========================================================================
# DECA Model (renderer-free)
# ===========================================================================

class DECAModel:
    """Minimal DECA wrapper: encoder + FLAME + detail decoder, no renderer."""

    def __init__(self, device="cuda"):
        from decalib.models.encoders import ResnetEncoder
        from decalib.models.FLAME import FLAME
        from decalib.models.decoders import Generator
        from decalib.utils import util
        from decalib.utils.config import cfg as deca_cfg
        from decalib.datasets import datasets
        from skimage.io import imread

        self.device = device
        self.util = util
        self.datasets = datasets
        self.cfg = deca_cfg
        model_cfg = deca_cfg.model

        n_param = (model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp +
                   model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light)
        n_detail = model_cfg.n_detail
        n_cond = model_cfg.n_exp + 3

        self.param_dict = {i: model_cfg.get("n_" + i) for i in model_cfg.param_list}

        print("  DECA 모델 로딩 중...")
        self.E_flame = ResnetEncoder(outsize=n_param).to(device)
        self.E_detail = ResnetEncoder(outsize=n_detail).to(device)
        self.flame = FLAME(model_cfg).to(device)
        self.D_detail = Generator(
            latent_dim=n_detail + n_cond, out_channels=1,
            out_scale=model_cfg.max_z, sample_mode="bilinear",
        ).to(device)

        checkpoint = torch.load(
            deca_cfg.pretrained_modelpath, map_location=device, weights_only=False
        )
        util.copy_state_dict(self.E_flame.state_dict(), checkpoint["E_flame"])
        util.copy_state_dict(self.E_detail.state_dict(), checkpoint["E_detail"])
        util.copy_state_dict(self.D_detail.state_dict(), checkpoint["D_detail"])
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

        self.dense_template = np.load(
            model_cfg.dense_template_path, allow_pickle=True, encoding="latin1"
        ).item()
        self.fixed_dis = np.load(model_cfg.fixed_displacement_path)
        face_eye_mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.0
        self.face_mask = face_eye_mask[:, :, 0]
        self.faces_np = self.flame.faces_tensor.cpu().numpy()

        # Precompute face-only vertex filtering for dense mesh
        # uv_face_mask.png defines face region in UV space (no ears, neck, scalp)
        face_only_uv = imread(model_cfg.face_mask_path).astype(np.float32) / 255.0
        if face_only_uv.ndim == 3:
            face_only_uv = face_only_uv[:, :, 0]

        vp_ids = self.dense_template["valid_pixel_ids"]
        uv_x = np.clip(self.dense_template["x_coords"][vp_ids].astype(int), 0, 255)
        uv_y = np.clip(self.dense_template["y_coords"][vp_ids].astype(int), 0, 255)
        self.is_face_vertex = face_only_uv[uv_y, uv_x] > 0.5

        dense_faces_all = self.dense_template["f"]
        all_in_face = self.is_face_vertex[dense_faces_all].all(axis=1)
        kept_faces = dense_faces_all[all_in_face]

        face_indices = np.where(self.is_face_vertex)[0]
        remap = -np.ones(len(self.is_face_vertex), dtype=np.int64)
        remap[face_indices] = np.arange(len(face_indices))
        self.face_only_dense_faces = remap[kept_faces]
        self.face_vertex_indices = face_indices

        n_total = len(self.is_face_vertex)
        n_face = len(face_indices)
        print(f"  Face-only filter: {n_face}/{n_total} vertices "
              f"({len(self.face_only_dense_faces)}/{len(dense_faces_all)} faces)")
        print("  DECA 로딩 완료!")

    def infer(self, img_path, out_dir=None):
        """Full pipeline: image → detail mesh + coarse mesh + landmarks.
        Returns dict with all outputs.
        """
        testdata = self.datasets.TestData(
            img_path, iscrop=True, face_detector="fan"
        )
        data = testdata[0]
        image = data["image"].to(self.device)[None, ...]
        tform_params = data["tform"].numpy()
        crop_rgb = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        with torch.no_grad():
            parameters = self.E_flame(image)
            detailcode = self.E_detail(image)

            code_dict = self._decompose(parameters)

            verts, landmarks2d, landmarks3d = self.flame(
                shape_params=code_dict["shape"],
                expression_params=code_dict["exp"],
                pose_params=code_dict["pose"],
            )

            uv_z = self.D_detail(torch.cat([
                code_dict["pose"][:, 3:], code_dict["exp"], detailcode
            ], dim=1))

            normals = self.util.vertex_normals(
                verts, self.flame.faces_tensor.expand(1, -1, -1)
            )

            lm2d = self.util.batch_orth_proj(landmarks2d, code_dict["cam"])[:, :, :2]
            lm2d[:, :, 1:] = -lm2d[:, :, 1:]

            trans_verts = self.util.batch_orth_proj(verts, code_dict["cam"])
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        verts_np = verts[0].cpu().numpy()
        normals_np = normals[0].cpu().numpy()
        lm2d_np = lm2d[0].cpu().numpy()
        trans_verts_np = trans_verts[0].cpu().numpy()

        displacement = (uv_z[0, 0].cpu().numpy() + self.fixed_dis) * self.face_mask

        dense_verts, _, dense_faces = self.util.upsample_mesh(
            verts_np, normals_np, self.faces_np,
            displacement,
            np.ones((256, 256, 3), dtype=np.float32) * 180,
            self.dense_template,
        )

        # Project dense vertices → sample colors from crop
        dense_verts_t = torch.tensor(
            dense_verts, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dense_trans = self.util.batch_orth_proj(dense_verts_t, code_dict["cam"])
            dense_trans[:, :, 1:] = -dense_trans[:, :, 1:]
        dense_2d = dense_trans[0].cpu().numpy()

        px = np.clip(((dense_2d[:, 0] + 1) / 2 * 224).astype(int), 0, 223)
        py = np.clip(((dense_2d[:, 1] + 1) / 2 * 224).astype(int), 0, 223)
        dense_colors = crop_rgb[py, px]

        # Project coarse vertices for composite overlay
        coarse_px = np.clip(((trans_verts_np[:, 0] + 1) / 2 * 224).astype(int), 0, 223)
        coarse_py = np.clip(((trans_verts_np[:, 1] + 1) / 2 * 224).astype(int), 0, 223)
        coarse_colors = crop_rgb[coarse_py, coarse_px]

        # Face-only dense mesh
        face_verts = dense_verts[self.face_vertex_indices]
        face_colors = dense_colors[self.face_vertex_indices]
        face_2d = dense_2d[self.face_vertex_indices]
        face_faces = self.face_only_dense_faces

        # Compute vertex normals for face-only mesh
        v0 = face_verts[face_faces[:, 0]]
        v1 = face_verts[face_faces[:, 1]]
        v2 = face_verts[face_faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
        face_normals = np.zeros_like(face_verts)
        np.add.at(face_normals, face_faces[:, 0], fn)
        np.add.at(face_normals, face_faces[:, 1], fn)
        np.add.at(face_normals, face_faces[:, 2], fn)
        face_normals /= (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            face_mesh = trimesh.Trimesh(
                vertices=face_verts, faces=face_faces,
                vertex_colors=face_colors, process=False,
            )
            face_mesh.export(str(out_dir / "detail.ply"))
            face_mesh.export(str(out_dir / "detail.obj"))

            full_mesh = trimesh.Trimesh(
                vertices=dense_verts, faces=dense_faces,
                vertex_colors=dense_colors, process=False,
            )
            full_mesh.export(str(out_dir / "full_head.ply"))

            coarse_mesh = trimesh.Trimesh(
                vertices=verts_np, faces=self.faces_np,
                vertex_colors=coarse_colors, process=False,
            )
            coarse_mesh.export(str(out_dir / "coarse.ply"))

        cam_params = code_dict["cam"][0].detach().cpu().numpy()

        return {
            "crop_rgb": crop_rgb,
            "lm2d": lm2d_np,
            "trans_verts": trans_verts_np,
            "dense_2d": face_2d,
            "dense_faces": face_faces,
            "dense_colors": face_colors,
            "face_normals": face_normals,
            "tform_params": tform_params,
            "cam_params": cam_params,
            "n_face_verts": len(face_verts),
            "n_face_faces": len(face_faces),
        }

    def _decompose(self, parameters):
        code_dict = {}
        start = 0
        for key in self.param_dict:
            end = start + int(self.param_dict[key])
            code_dict[key] = parameters[:, start:end]
            start = end
            if key == "light":
                code_dict[key] = code_dict[key].reshape(
                    code_dict[key].shape[0], 9, 3
                )
        return code_dict


# ===========================================================================
# Rendering (OpenCV painter's algorithm for detail mesh)
# ===========================================================================

def render_painter(base_img, verts_2d, faces, vertex_colors):
    """Render mesh onto image using painter's algorithm (back-to-front)."""
    h, w = base_img.shape[:2]
    overlay = base_img.copy()

    face_z = verts_2d[faces, 2].mean(axis=1) if verts_2d.shape[1] >= 3 else np.zeros(len(faces))
    order = np.argsort(-face_z)

    for i in order:
        f = faces[i]
        pts = verts_2d[f, :2]
        pts_px = ((pts + 1) / 2 * np.array([w, h])).astype(np.int32)
        color = vertex_colors[f].mean(axis=0).astype(int).tolist()
        cv2.fillConvexPoly(overlay, pts_px, color)

    return overlay


def create_composite(crop_rgb, dense_2d, dense_faces, dense_colors, lm2d):
    """Create composite: [crop | 3D overlay | landmarks]"""
    h, w = crop_rgb.shape[:2]

    panel1 = crop_rgb.copy()
    panel2 = render_painter(crop_rgb, dense_2d, dense_faces, dense_colors)

    panel3 = panel2.copy()
    lm_px = ((lm2d + 1) / 2 * np.array([w, h])).astype(int)
    for pt in lm_px:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(panel3, (x, y), 2, (255, 0, 0), -1)

    return np.hstack([panel1, panel2, panel3])


# ===========================================================================
# Rasterizer (normal + depth + mask in crop space)
# ===========================================================================

def rasterize_geometry(verts_2d, normals, depths, faces, img_size=224):
    """Rasterize normals/depth/mask with z-buffer in crop pixel space."""
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
        denom = (v1[1]-v2[1])*(v0[0]-v2[0]) + (v2[0]-v1[0])*(v0[1]-v2[1])
        if abs(denom) < 1e-8:
            continue
        pxs = np.arange(x_min, x_max + 1)
        pys = np.arange(y_min, y_max + 1)
        gx, gy = np.meshgrid(pxs, pys)
        gx, gy = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)
        w0 = ((v1[1]-v2[1])*(gx-v2[0]) + (v2[0]-v1[0])*(gy-v2[1])) / denom
        w1 = ((v2[1]-v0[1])*(gx-v2[0]) + (v0[0]-v2[0])*(gy-v2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        idx = np.where(inside)[0]
        px_in, py_in = gx[idx].astype(np.int32), gy[idx].astype(np.int32)
        w0i, w1i, w2i = w0[idx], w1[idx], w2[idx]
        zi = w0i * z0 + w1i * z1 + w2i * z2
        for i in range(len(idx)):
            px, py = px_in[i], py_in[i]
            if zi[i] < depth_map[py, px]:
                depth_map[py, px] = zi[i]
                normal_map[py, px] = w0i[i]*n0 + w1i[i]*n1 + w2i[i]*n2
                mask[py, px] = 1.0

    depth_map[mask == 0] = 0.0
    return normal_map, depth_map, mask


# ===========================================================================
# Silicone material transfer
# ===========================================================================

def apply_silicone(orig_rgb, normal_map, depth_map, mask):
    """Apply silicone mask effect on original image using 3D normals."""
    result = orig_rgb.copy()
    m3 = mask[:, :, np.newaxis]

    smooth1 = cv2.GaussianBlur(result, (15, 15), 5.0)
    smooth2 = cv2.bilateralFilter(
        (smooth1 * 255).astype(np.uint8), 9, 75, 75
    ).astype(np.float32) / 255.0
    result = result * (1 - m3) + (result * 0.15 + smooth2 * 0.85) * m3

    local_mean = cv2.GaussianBlur(result, (31, 31), 12.0)
    deviation = result - local_mean
    result = result * (1 - m3) + (local_mean + deviation * 0.3) * m3

    gray = np.mean(result, axis=2, keepdims=True)
    result = result * (1 - m3) + (result * 0.65 + gray * 0.35) * m3

    fc = result.copy()
    fc[:, :, 0] *= 1.06
    fc[:, :, 1] *= 1.02
    fc[:, :, 2] *= 0.88
    result = result * (1 - m3) + fc * m3

    fp = result[mask > 0.5]
    mean_rgb = np.mean(fp.reshape(-1, 3), axis=0) if len(fp) > 0 else np.array([0.5, 0.5, 0.5])
    result = result * (1 - m3) + (result * 0.7 + mean_rgb * 0.3) * m3

    half = np.array([0.0, 0.0, 1.0])
    ndh = np.clip(np.sum(normal_map * half, axis=2), 0, 1)
    spec = (np.power(ndh, 4) * 0.12 + np.power(ndh, 20) * 0.08) * mask
    result = result + spec[:, :, np.newaxis] * m3

    result = result * (1 - m3) + (result * 1.05) * m3
    return np.clip(result, 0, 1)


def process_silicone(result, img_path):
    """Full silicone pipeline: rasterize normals → warp to original → apply silicone."""
    from skimage.transform import SimilarityTransform

    crop_size = 224
    dense_2d = result["dense_2d"]
    dense_faces = result["dense_faces"]
    face_normals = result["face_normals"]
    lm2d = result["lm2d"]
    tform_params = result["tform_params"]

    # 1. Convert normalized coords to crop pixel space
    verts_px = np.zeros((len(dense_2d), 2), dtype=np.float64)
    verts_px[:, 0] = (dense_2d[:, 0] + 1) / 2 * crop_size
    verts_px[:, 1] = (dense_2d[:, 1] + 1) / 2 * crop_size
    depths = dense_2d[:, 2] if dense_2d.shape[1] >= 3 else np.zeros(len(dense_2d))

    # 2. Rasterize normals/depth/mask in 224x224 crop
    normal_crop, depth_crop, mask_crop = rasterize_geometry(
        verts_px, face_normals, depths, dense_faces, img_size=crop_size
    )

    # 3. Warp from crop to original image
    tform = SimilarityTransform()
    tform.params = tform_params.astype(np.float64)
    M_orig2crop = tform.params[:2]  # maps original → crop

    orig_bgr = cv2.imread(str(img_path))
    h, w = orig_bgr.shape[:2]

    normal_orig = cv2.warpAffine(normal_crop, M_orig2crop, (w, h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth_orig = cv2.warpAffine(depth_crop, M_orig2crop, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_orig = cv2.warpAffine(mask_crop, M_orig2crop, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 4. Soft mask with erosion + blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_soft = cv2.erode(mask_orig, kernel, iterations=2)
    mask_soft = cv2.GaussianBlur(mask_soft, (21, 21), 7.0)

    # 5. Eye/mouth holes using 68 landmarks
    lm_crop_px = (lm2d + 1) / 2 * crop_size
    lm_orig = tform.inverse(lm_crop_px).astype(np.int32)

    eye_l = lm_orig[36:42].copy()
    eye_r = lm_orig[42:48].copy()
    mouth = lm_orig[48:60].copy()

    for pts in [eye_l, eye_r]:
        c = pts.mean(axis=0)
        pts[:] = ((pts - c) * 1.4 + c).astype(np.int32)
    mc = mouth.mean(axis=0)
    mouth = ((mouth - mc) * 1.2 + mc).astype(np.int32)

    hole = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(hole, [eye_l], 1.0)
    cv2.fillPoly(hole, [eye_r], 1.0)
    cv2.fillPoly(hole, [mouth], 1.0)
    hole = cv2.GaussianBlur(hole, (15, 15), 5.0)
    mask_soft = np.clip(mask_soft - hole, 0, 1)

    # 6. Apply silicone material
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    silicone_rgb = apply_silicone(orig_rgb, normal_orig, depth_orig, mask_soft)
    silicone_bgr = cv2.cvtColor((silicone_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return silicone_bgr


# ===========================================================================
# Blender PBR Rendering + Composite
# ===========================================================================

BLENDER_PATH = str(ROOT.parent / "tools" / "blender-4.2.13-linux-x64" / "blender")
RENDER_SCRIPT = str(ROOT / "scripts" / "render_blender.py")
MATERIAL_NAMES = ["silicone", "latex", "resin"]


def blender_render_and_composite(result, img_path, out_dir):
    """Blender PBR rendering in DECA crop space → composite onto original."""
    from composite_blender import composite_render

    cam_params = result["cam_params"]
    obj_path = out_dir / "detail.obj"
    original_bgr = cv2.imread(str(img_path))

    cmd = [
        BLENDER_PATH, "--background", "--python", RENDER_SCRIPT, "--",
        "--obj", str(obj_path),
        "--material", "all",
        "--output_dir", str(out_dir),
        "--samples", "128",
        "--cam_params", str(cam_params[0]), str(cam_params[1]), str(cam_params[2]),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"  Blender error: {proc.stderr[-500:]}")
        return

    tform_params = result["tform_params"]
    lm2d = result["lm2d"]

    for mat_name in MATERIAL_NAMES:
        render_path = out_dir / f"{mat_name}_render.png"
        if not render_path.exists():
            print(f"    {mat_name}_render.png MISSING")
            continue
        render_rgba = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
        comp = composite_render(original_bgr, render_rgba, tform_params, lm2d)
        cv2.imwrite(str(out_dir / f"{mat_name}_composite.png"), comp)
        print(f"    {mat_name}_composite.png OK")


# ===========================================================================
# HTML Template
# ===========================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DECA Face Reconstruction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
            padding: 32px;
            text-align: center;
            border-bottom: 1px solid #333;
        }
        .header h1 { font-size: 28px; color: #fff; margin-bottom: 8px; }
        .header p { color: #888; font-size: 14px; }
        .header .badge {
            display: inline-block;
            background: #e94560;
            color: #fff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 32px; }
        .upload-section {
            background: #1a1a1a;
            border: 2px dashed #444;
            border-radius: 16px;
            padding: 48px;
            text-align: center;
            margin-bottom: 32px;
            transition: border-color 0.3s;
        }
        .upload-section:hover { border-color: #e94560; }
        .upload-section label {
            display: inline-block;
            background: #e94560;
            color: #fff;
            padding: 14px 32px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
        }
        .upload-section label:hover { background: #d63351; }
        .upload-section input[type="file"] { display: none; }
        .upload-section .hint { margin-top: 12px; color: #666; font-size: 13px; }
        .processing { text-align: center; padding: 60px; display: none; }
        .spinner {
            width: 48px; height: 48px;
            border: 4px solid #333;
            border-top: 4px solid #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .section {
            background: #1a1a1a;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .section h2 { margin-bottom: 16px; font-size: 20px; color: #fff; }
        #viewer3d {
            width: 100%;
            height: 500px;
            border-radius: 12px;
            border: 1px solid #333;
            overflow: hidden;
            background: #111;
        }
        .viewer-controls {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            justify-content: center;
        }
        .viewer-controls button {
            background: #333;
            color: #ccc;
            border: 1px solid #555;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }
        .viewer-controls button:hover { background: #444; }
        .composite-img {
            width: 100%;
            max-width: 672px;
            border-radius: 8px;
            border: 1px solid #333;
            display: block;
            margin: 0 auto;
        }
        .composite-labels {
            display: flex;
            max-width: 672px;
            margin: 8px auto 0;
            text-align: center;
        }
        .composite-labels span { flex: 1; color: #888; font-size: 13px; }
        .download-btn {
            display: inline-block;
            color: #fff;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            transition: opacity 0.3s;
            margin: 4px;
        }
        .download-btn:hover { opacity: 0.85; }
        .actions { text-align: center; margin-top: 16px; }
        .stats {
            display: flex;
            gap: 16px;
            margin-top: 16px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .stat-item {
            background: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px 20px;
            text-align: center;
        }
        .stat-item .value { font-size: 24px; font-weight: 700; color: #e94560; }
        .stat-item .label { font-size: 12px; color: #888; margin-top: 4px; }
        .pipeline-info {
            background: #111;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
        }
        .pipeline-info h3 { color: #e94560; margin-bottom: 12px; }
        .pipeline-info ol { padding-left: 20px; }
        .pipeline-info li { margin-bottom: 6px; color: #999; font-size: 13px; }
        .pipeline-info li span { color: #ccc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>DECA Face Reconstruction <span class="badge">Detail Mesh</span></h1>
        <p>FLAME + Displacement Map → 59K vertices detail mesh</p>
    </div>
    <div class="container">
        <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
            <div class="upload-section">
                <label for="fileInput">얼굴 사진 업로드</label>
                <input type="file" id="fileInput" name="image" accept="image/jpeg,image/png"
                       onchange="submitForm()">
                <p class="hint">JPG / PNG 형식. 정면 얼굴이 포함된 이미지를 업로드하세요.</p>
            </div>
        </form>
        <div class="processing" id="processing">
            <div class="spinner"></div>
            <p>3D 복원 중... (약 5~15초)</p>
            <p style="color:#666; font-size:13px; margin-top:8px;">
                FAN 랜드마크 → FLAME 복원 → Detail Displacement → 메시 생성
            </p>
        </div>

        {% if result_id %}
        <div class="section">
            <h2>3D Detail Mesh 뷰어</h2>
            <div id="viewer3d"></div>
            <div class="viewer-controls">
                <button onclick="resetCamera()">카메라 리셋</button>
                <button onclick="toggleWireframe()">와이어프레임</button>
                <button onclick="toggleLight()">조명 토글</button>
                <button onclick="toggleDetail()">Coarse/Detail 전환</button>
                <button onclick="toggleFullHead()">Full Head 토글</button>
            </div>
        </div>

        <div class="section">
            <h2>3D 복원 결과</h2>
            <img class="composite-img" src="/image/{{ result_id }}/composite" alt="Composite">
            <div class="composite-labels">
                <span>Input Crop</span>
                <span>Detail Mesh Overlay</span>
                <span>68 Landmarks</span>
            </div>
            {% if stats %}
            <div class="stats">
                <div class="stat-item">
                    <div class="value">{{ stats.detail_verts }}</div>
                    <div class="label">Detail 정점</div>
                </div>
                <div class="stat-item">
                    <div class="value">{{ stats.detail_faces }}</div>
                    <div class="label">Detail 면</div>
                </div>
                <div class="stat-item">
                    <div class="value">{{ stats.time }}s</div>
                    <div class="label">처리 시간</div>
                </div>
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>3D Mask Material Synthesis <span class="badge">Blender PBR</span></h2>
            <p style="color:#aaa; margin-bottom:12px;">Blender Cycles PBR 렌더링 → 원본 합성 (Principled BSDF + SSS + 스튜디오 조명)</p>
            <div style="display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap; justify-content:center;">
                <div style="text-align:center;">
                    <img src="/image/{{ result_id }}/original" alt="원본" style="max-width:240px; width:100%; border-radius:8px;">
                    <div style="color:#888; font-size:13px; margin-top:4px;">원본</div>
                </div>
                <div style="text-align:center;">
                    <img src="/image/{{ result_id }}/silicone_composite" alt="실리콘" style="max-width:240px; width:100%; border-radius:8px;">
                    <div style="color:#888; font-size:13px; margin-top:4px;">실리콘 (Silicone)</div>
                </div>
                <div style="text-align:center;">
                    <img src="/image/{{ result_id }}/latex_composite" alt="라텍스" style="max-width:240px; width:100%; border-radius:8px;">
                    <div style="color:#888; font-size:13px; margin-top:4px;">라텍스 (Latex)</div>
                </div>
                <div style="text-align:center;">
                    <img src="/image/{{ result_id }}/resin_composite" alt="레진" style="max-width:240px; width:100%; border-radius:8px;">
                    <div style="color:#888; font-size:13px; margin-top:4px;">레진 (Resin)</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>다운로드</h2>
            <div class="actions">
                <a class="download-btn" style="background:#e94560;"
                   href="/file/{{ result_id }}/detail.obj">Detail Mesh (.obj)</a>
                <a class="download-btn" style="background:#6a4aff;"
                   href="/file/{{ result_id }}/detail.ply">Detail Mesh (.ply)</a>
                <a class="download-btn" style="background:#4a6aff;"
                   href="/file/{{ result_id }}/coarse.ply">Coarse Mesh (.ply)</a>
                <a class="download-btn" style="background:#ff6b4a;"
                   href="/file/{{ result_id }}/full_head.ply">Full Head (.ply)</a>
                <a class="download-btn" style="background:#ff9f43;"
                   href="/file/{{ result_id }}/silicone_composite.png">실리콘 합성</a>
                <a class="download-btn" style="background:#f0a040;"
                   href="/file/{{ result_id }}/latex_composite.png">라텍스 합성</a>
                <a class="download-btn" style="background:#d0903a;"
                   href="/file/{{ result_id }}/resin_composite.png">레진 합성</a>
                <a class="download-btn" style="background:#2d8f4e;"
                   href="/file/{{ result_id }}/composite.png">3D 복원 이미지</a>
                <a class="download-btn" style="background:#555;"
                   href="/file/{{ result_id }}/original.jpg">원본 이미지</a>
            </div>
        </div>
        {% endif %}

        {% if error %}
        <div class="section" style="border: 1px solid #ff4444;">
            <h2 style="color: #ff4444;">오류</h2>
            <p>{{ error }}</p>
        </div>
        {% endif %}

        <div class="pipeline-info">
            <h3>DECA Pipeline</h3>
            <ol>
                <li><span>FAN 랜드마크</span> — face_alignment으로 68-point 랜드마크 검출</li>
                <li><span>DECA Encoder</span> — E_flame (FLAME 파라미터) + E_detail (디테일 코드) 추출</li>
                <li><span>FLAME Decode</span> — shape + expression + pose → coarse mesh (5K vertices)</li>
                <li><span>Detail Decoder</span> — D_detail → UV displacement map (주름, 모공)</li>
                <li><span>Mesh Upsample</span> — displacement + dense template → detail mesh (59K vertices)</li>
                <li><span>Vertex Color</span> — orthographic projection으로 원본 이미지에서 색상 추출</li>
                <li><span>Face Filter</span> — UV face mask로 얼굴 영역만 추출 (21K vertices)</li>
                <li><span>Blender PBR</span> — Cycles GPU 렌더링 (실리콘/라텍스/레진) + 원본 이미지에 합성</li>
            </ol>
        </div>
    </div>

    {% if result_id %}
    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    <script type="module">
    import * as THREE from 'three';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    const container = document.getElementById('viewer3d');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(
        45, container.clientWidth / container.clientHeight, 0.01, 100
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0, 1, 2);
    scene.add(dirLight);

    let detailObj = null, coarseObj = null, fullHeadObj = null;
    let wireframeOn = false, lightOn = true, showingDetail = true;
    let meshCenter = new THREE.Vector3();
    const loader = new PLYLoader();

    function loadMesh(url, visible, cb) {
        loader.load(url, function(geometry) {
            geometry.computeVertexNormals();
            const material = new THREE.MeshStandardMaterial({
                vertexColors: true, roughness: 0.5, metalness: 0.0,
            });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.visible = visible;
            scene.add(mesh);
            cb(mesh, geometry);
        });
    }

    loadMesh('/mesh/{{ result_id }}/detail', true, function(mesh, geo) {
        detailObj = mesh;
        geo.computeBoundingBox();
        geo.boundingBox.getCenter(meshCenter);
        detailObj.position.sub(meshCenter.clone());
        const size = new THREE.Vector3();
        geo.boundingBox.getSize(size);
        camera.position.set(0, 0, Math.max(size.x, size.y, size.z) * 2.5);
        controls.target.set(0, 0, 0);
        controls.update();
    });

    loadMesh('/mesh/{{ result_id }}/coarse', false, function(mesh) {
        coarseObj = mesh;
        coarseObj.position.sub(meshCenter.clone());
    });

    loadMesh('/mesh/{{ result_id }}/full_head', false, function(mesh) {
        fullHeadObj = mesh;
        fullHeadObj.position.sub(meshCenter.clone());
    });

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        const w = container.clientWidth, h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    window.resetCamera = function() {
        const obj = showingDetail ? detailObj : coarseObj;
        if (obj) {
            const box = new THREE.Box3().setFromObject(obj);
            const size = new THREE.Vector3();
            box.getSize(size);
            camera.position.set(0, 0, Math.max(size.x, size.y, size.z) * 2.5);
            controls.target.set(0, 0, 0);
            controls.update();
        }
    };
    window.toggleWireframe = function() {
        wireframeOn = !wireframeOn;
        if (detailObj) detailObj.material.wireframe = wireframeOn;
        if (coarseObj) coarseObj.material.wireframe = wireframeOn;
    };
    window.toggleLight = function() {
        lightOn = !lightOn;
        dirLight.visible = lightOn;
    };
    window.toggleDetail = function() {
        showingDetail = !showingDetail;
        if (fullHeadObj) fullHeadObj.visible = false;
        if (detailObj) detailObj.visible = showingDetail;
        if (coarseObj) coarseObj.visible = !showingDetail;
    };
    window.toggleFullHead = function() {
        if (fullHeadObj) {
            const show = !fullHeadObj.visible;
            fullHeadObj.visible = show;
            if (detailObj) detailObj.visible = !show;
            if (coarseObj) coarseObj.visible = false;
        }
    };
    </script>
    {% endif %}

    <script>
    function submitForm() {
        document.getElementById('processing').style.display = 'block';
        document.getElementById('uploadForm').submit();
    }
    </script>
</body>
</html>
"""


# ===========================================================================
# Flask App
# ===========================================================================

app = Flask(__name__)
deca_model = None


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE, result_id=None, error=None, stats=None
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return render_template_string(
            HTML_TEMPLATE, result_id=None,
            error="파일이 선택되지 않았습니다.", stats=None
        )
    file = request.files["image"]
    if file.filename == "":
        return render_template_string(
            HTML_TEMPLATE, result_id=None,
            error="파일이 선택되지 않았습니다.", stats=None
        )

    result_id = str(uuid.uuid4())[:8]
    work_dir = UPLOAD_DIR / result_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RESULT_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return render_template_string(
            HTML_TEMPLATE, result_id=None,
            error="JPG 또는 PNG 파일만 지원합니다.", stats=None
        )

    img_path = work_dir / f"input{ext}"
    file.save(str(img_path))

    try:
        t0 = time.time()
        print(f"\n[{result_id}] 처리 시작: {file.filename}")

        print(f"  [1/2] DECA Inference...")
        result = deca_model.infer(str(img_path), out_dir=out_dir)

        print(f"  [2/3] 2D Composite 생성...")
        composite_rgb = create_composite(
            result["crop_rgb"],
            result["dense_2d"],
            result["dense_faces"],
            result["dense_colors"],
            result["lm2d"],
        )
        composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "composite.png"), composite_bgr)

        orig_bgr = cv2.imread(str(img_path))
        cv2.imwrite(str(out_dir / "original.jpg"), orig_bgr)

        print(f"  [3/3] Blender PBR 렌더링 + 합성 (3종)...")
        blender_render_and_composite(result, str(img_path), out_dir)

        elapsed = round(time.time() - t0, 1)
        stats = {
            "detail_verts": f"{result['n_face_verts']:,}",
            "detail_faces": f"{result['n_face_faces']:,}",
            "time": elapsed,
        }
        print(f"  [완료] {result_id} ({elapsed}s)")

        return render_template_string(
            HTML_TEMPLATE, result_id=result_id, error=None, stats=stats
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template_string(
            HTML_TEMPLATE, result_id=None,
            error=f"처리 중 오류: {str(e)}", stats=None
        )


@app.route("/image/<result_id>/<img_type>")
def serve_image(result_id, img_type):
    type_map = {
        "original": "original.jpg", "composite": "composite.png",
        "silicone_composite": "silicone_composite.png",
        "latex_composite": "latex_composite.png",
        "resin_composite": "resin_composite.png",
    }
    if img_type not in type_map:
        return "Not found", 404
    path = RESULT_DIR / result_id / type_map[img_type]
    if not path.exists():
        return "Not found", 404
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    if request.args.get("download"):
        return send_file(
            str(path), as_attachment=True,
            download_name=f"{img_type}_{result_id}{path.suffix}"
        )
    return send_file(str(path), mimetype=mime)


@app.route("/mesh/<result_id>/<mesh_type>")
def serve_mesh(result_id, mesh_type):
    name_map = {"detail": "detail.ply", "coarse": "coarse.ply", "full_head": "full_head.ply"}
    if mesh_type not in name_map:
        return "Not found", 404
    path = RESULT_DIR / result_id / name_map[mesh_type]
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype="application/octet-stream")


@app.route("/file/<result_id>/<filename>")
def serve_file(result_id, filename):
    ALLOWED = {
        "detail.obj", "detail.ply", "coarse.ply", "full_head.ply",
        "composite.png", "original.jpg",
        "silicone_composite.png", "latex_composite.png", "resin_composite.png",
    }
    if filename not in ALLOWED:
        return "Not found", 404
    path = RESULT_DIR / result_id / filename
    if not path.exists():
        return "Not found", 404
    return send_file(
        str(path), as_attachment=True,
        download_name=f"{result_id}_{filename}"
    )


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DECA Face Reconstruction 초기화 중...")
    print("=" * 60)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/1] DECA 모델 로딩...")
    deca_model = DECAModel(device="cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  서버 시작: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
