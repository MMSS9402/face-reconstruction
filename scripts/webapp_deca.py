"""
Flask 웹앱: DECA 기반 2장 입력 3D mask 합성
renderer 없이 DECA 인코더 + FLAME + Detail Decoder만 사용

사용법:
    conda run -n deca-env python scripts/webapp_deca.py
    → http://localhost:5000
"""

import sys
import os
import uuid
import time
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
        """Full pipeline: image -> detail mesh + coarse mesh + landmarks."""
        image, tform_params, crop_rgb = self._load_image_data(img_path)
        code_dict, uv_z, normals, lm2d, trans_verts, verts = self._encode_crop(image)

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

        coarse_px = np.clip(((trans_verts_np[:, 0] + 1) / 2 * 224).astype(int), 0, 223)
        coarse_py = np.clip(((trans_verts_np[:, 1] + 1) / 2 * 224).astype(int), 0, 223)
        coarse_colors = crop_rgb[coarse_py, coarse_px]

        face_verts = dense_verts[self.face_vertex_indices]
        face_colors = dense_colors[self.face_vertex_indices]
        face_2d = dense_2d[self.face_vertex_indices]
        face_faces = self.face_only_dense_faces

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

    def extract_face_reference(self, img_path):
        """Extract crop, transform, and landmarks without mesh export."""
        image, tform_params, crop_rgb = self._load_image_data(img_path)
        code_dict, _, _, lm2d, _, _ = self._encode_crop(image)
        cam_params = code_dict["cam"][0].detach().cpu().numpy()
        return {
            "crop_rgb": crop_rgb,
            "lm2d": lm2d[0].cpu().numpy(),
            "tform_params": tform_params,
            "cam_params": cam_params,
        }

    def _load_image_data(self, img_path):
        testdata = self.datasets.TestData(
            img_path, iscrop=True, face_detector="fan"
        )
        data = testdata[0]
        image = data["image"].to(self.device)[None, ...]
        tform_params = data["tform"].numpy()
        crop_rgb = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return image, tform_params, crop_rgb

    def _encode_crop(self, image):
        with torch.no_grad():
            parameters = self.E_flame(image)
            detailcode = self.E_detail(image)

            code_dict = self._decompose(parameters)

            verts, landmarks2d, _ = self.flame(
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

        return code_dict, uv_z, normals, lm2d, trans_verts, verts

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


def render_painter_rgba(verts_2d, faces, vertex_colors, image_size=224):
    """Render detail mesh with alpha in crop space."""
    rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    alpha = np.zeros((image_size, image_size), dtype=np.uint8)

    face_z = verts_2d[faces, 2].mean(axis=1) if verts_2d.shape[1] >= 3 else np.zeros(len(faces))
    order = np.argsort(-face_z)

    for i in order:
        f = faces[i]
        pts = verts_2d[f, :2]
        pts_px = ((pts + 1) / 2 * np.array([image_size, image_size])).astype(np.int32)
        color = vertex_colors[f].mean(axis=0).astype(int).tolist()
        cv2.fillConvexPoly(rgb, pts_px, color)
        cv2.fillConvexPoly(alpha, pts_px, 255)

    return np.dstack([rgb, alpha])


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
# Blender Rendering + Target Composite
# ===========================================================================

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def blender_render_and_composite(source_result, target_result, target_img_path, out_dir):
    """Composite the source detail mesh overlay onto the target face."""
    from composite_blender import composite_render

    target_bgr = cv2.imread(str(target_img_path))
    if target_bgr is None:
        raise RuntimeError("타깃 이미지를 읽지 못했습니다.")

    render_rgba = render_painter_rgba(
        source_result["dense_2d"],
        source_result["dense_faces"],
        source_result["dense_colors"],
    )
    render_bgra = cv2.cvtColor(render_rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(out_dir / "mask_render.png"), render_bgra)

    comp = composite_render(
        target_bgr,
        render_bgra,
        target_result["tform_params"],
        target_result["lm2d"],
        source_lm2d=source_result["lm2d"],
    )
    cv2.imwrite(str(out_dir / "mask_composite.png"), comp)


# ===========================================================================
# HTML Template
# ===========================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DECA Face Mask Composer</title>
    <style>
        body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 0; }
        .wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
        .section { background: #1b1b1b; border-radius: 14px; padding: 20px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
        .card { background: #151515; border: 1px solid #333; border-radius: 12px; padding: 12px; text-align: center; }
        .card img { width: 100%; border-radius: 8px; border: 1px solid #333; }
        .label { margin-top: 8px; color: #aaa; font-size: 13px; }
        .upload-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .upload-box { background: #141414; border: 1px dashed #444; border-radius: 12px; padding: 20px; text-align: center; }
        .upload-box input[type=file] { width: 100%; margin-top: 10px; }
        .btn { display: inline-block; padding: 10px 18px; border-radius: 8px; text-decoration: none; font-weight: 700; border: 0; cursor: pointer; }
        .btn-primary { background: #2d8f4e; color: white; }
        .btn-link { background: #2c5fd5; color: white; margin: 4px; }
        .stats { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin-top: 14px; }
        .stat { background: #121212; border: 1px solid #333; border-radius: 10px; padding: 10px 16px; text-align: center; }
        #viewer3d { width: 100%; height: 480px; border-radius: 10px; border: 1px solid #333; overflow: hidden; background: #0f0f0f; }
        .viewer-controls { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 12px; }
        .viewer-controls button { background: #333; color: #ddd; border: 1px solid #555; padding: 8px 14px; border-radius: 6px; cursor: pointer; }
        .processing { display: none; text-align: center; padding: 32px 0; color: #bbb; }
        .wide { width: 100%; max-width: 680px; display: block; margin: 0 auto; border-radius: 8px; border: 1px solid #333; }
        .labels { display: flex; max-width: 680px; margin: 8px auto 0; }
        .labels span { flex: 1; text-align: center; color: #888; font-size: 13px; }
        h1, h2 { margin-top: 0; }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="section">
            <h1>DECA Face Mask Composer</h1>
            <p>첫 번째 얼굴에서 3D mask를 만들고, 두 번째 얼굴에 landmark 기준으로 합성합니다.</p>
        </div>

        <form class="section" id="uploadForm" action="/upload" method="post" enctype="multipart/form-data" onsubmit="showProcessing()">
            <h2>입력 이미지</h2>
            <div class="upload-grid">
                <div class="upload-box">
                    <div><strong>Source Image</strong></div>
                    <div class="label">3D mask를 추출할 얼굴 이미지</div>
                    <input type="file" name="source_image" accept="image/jpeg,image/png" required>
                </div>
                <div class="upload-box">
                    <div><strong>Target Image</strong></div>
                    <div class="label">추출한 3D mask를 합성할 원본 이미지</div>
                    <input type="file" name="target_image" accept="image/jpeg,image/png" required>
                </div>
            </div>
            <div style="text-align:center; margin-top:16px;"><button class="btn btn-primary" type="submit">3D Mask 생성 및 합성</button></div>
        </form>

        <div class="processing" id="processing">Source 복원 -> Detail Mesh Render -> Target landmark 정렬 -> Target 합성</div>

        {% if error %}
        <div class="section"><h2>오류</h2><p>{{ error }}</p></div>
        {% endif %}

        {% if result_id %}
        <div class="section">
            <h2>입력 이미지 비교</h2>
            <div class="grid">
                <div class="card"><img src="/image/{{ result_id }}/source_original" alt="source"><div class="label">Source Image</div></div>
                <div class="card"><img src="/image/{{ result_id }}/target_original" alt="target"><div class="label">Target Image</div></div>
            </div>
        </div>

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
            <h2>Source 3D 복원 결과</h2>
            <img class="wide" src="/image/{{ result_id }}/source_reconstruction" alt="source reconstruction">
            <div class="labels"><span>Input Crop</span><span>Detail Mesh Overlay</span><span>68 Landmarks</span></div>
            {% if stats %}
            <div class="stats">
                <div class="stat"><div>{{ stats.detail_verts }}</div><div class="label">Detail 정점</div></div>
                <div class="stat"><div>{{ stats.detail_faces }}</div><div class="label">Detail 면</div></div>
                <div class="stat"><div>{{ stats.time }}s</div><div class="label">처리 시간</div></div>
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>Target 합성 결과</h2>
            <div class="grid">
                <div class="card"><img src="/image/{{ result_id }}/target_original" alt="target original"><div class="label">Target Original</div></div>
                <div class="card"><img src="/image/{{ result_id }}/mask_render" alt="detail mesh render"><div class="label">Detail Mesh Render</div></div>
                <div class="card"><img src="/image/{{ result_id }}/mask_composite" alt="mask composite"><div class="label">Mask Composite</div></div>
            </div>
        </div>

        <div class="section">
            <h2>다운로드</h2>
            <div style="text-align:center;">
                <a class="btn btn-link" href="/file/{{ result_id }}/detail.obj">detail.obj</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/detail.ply">detail.ply</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/coarse.ply">coarse.ply</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/full_head.ply">full_head.ply</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/source_reconstruction.png">source_reconstruction.png</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/mask_render.png">mask_render.png</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/mask_composite.png">mask_composite.png</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/source_original.jpg">source_original.jpg</a>
                <a class="btn btn-link" href="/file/{{ result_id }}/target_original.jpg">target_original.jpg</a>
            </div>
        </div>
        {% endif %}
    </div>

    {% if result_id %}
    <script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
    <script type="module">
    import * as THREE from 'three';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    const container = document.getElementById('viewer3d');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8); dirLight.position.set(0, 1, 2); scene.add(dirLight);
    let detailObj = null, coarseObj = null, fullHeadObj = null;
    let wireframeOn = false, lightOn = true, showingDetail = true;
    let meshCenter = new THREE.Vector3();
    const loader = new PLYLoader();
    function loadMesh(url, visible, cb) {
        loader.load(url, function(geometry) {
            geometry.computeVertexNormals();
            const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.5, metalness: 0.0 });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.visible = visible;
            scene.add(mesh);
            cb(mesh, geometry);
        });
    }
    loadMesh('/mesh/{{ result_id }}/detail', true, function(mesh, geo) {
        detailObj = mesh; geo.computeBoundingBox(); geo.boundingBox.getCenter(meshCenter); detailObj.position.sub(meshCenter.clone());
        const size = new THREE.Vector3(); geo.boundingBox.getSize(size); camera.position.set(0, 0, Math.max(size.x, size.y, size.z) * 2.5); controls.target.set(0, 0, 0); controls.update();
    });
    loadMesh('/mesh/{{ result_id }}/coarse', false, function(mesh) { coarseObj = mesh; coarseObj.position.sub(meshCenter.clone()); });
    loadMesh('/mesh/{{ result_id }}/full_head', false, function(mesh) { fullHeadObj = mesh; fullHeadObj.position.sub(meshCenter.clone()); });
    function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
    animate();
    window.addEventListener('resize', () => { const w = container.clientWidth, h = container.clientHeight; camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); });
    window.resetCamera = function() { const obj = showingDetail ? detailObj : coarseObj; if (obj) { const box = new THREE.Box3().setFromObject(obj); const size = new THREE.Vector3(); box.getSize(size); camera.position.set(0, 0, Math.max(size.x, size.y, size.z) * 2.5); controls.target.set(0, 0, 0); controls.update(); } };
    window.toggleWireframe = function() { wireframeOn = !wireframeOn; if (detailObj) detailObj.material.wireframe = wireframeOn; if (coarseObj) coarseObj.material.wireframe = wireframeOn; };
    window.toggleLight = function() { lightOn = !lightOn; dirLight.visible = lightOn; };
    window.toggleDetail = function() { showingDetail = !showingDetail; if (fullHeadObj) fullHeadObj.visible = false; if (detailObj) detailObj.visible = showingDetail; if (coarseObj) coarseObj.visible = !showingDetail; };
    window.toggleFullHead = function() { if (fullHeadObj) { const show = !fullHeadObj.visible; fullHeadObj.visible = show; if (detailObj) detailObj.visible = !show; if (coarseObj) coarseObj.visible = false; } };
    </script>
    {% endif %}
    <script>function showProcessing(){document.getElementById('processing').style.display='block';}</script>
</body>
</html>
"""


# ===========================================================================
# Flask App
# ===========================================================================

app = Flask(__name__)
deca_model = None


def render_page(result_id=None, error=None, stats=None):
    return render_template_string(
        HTML_TEMPLATE, result_id=result_id, error=error, stats=stats
    )


@app.route("/")
def index():
    return render_page()


def validate_upload_file(file, label):
    if file is None or file.filename == "":
        raise ValueError(f"{label} 파일이 선택되지 않았습니다.")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError(f"{label}는 JPG 또는 PNG 파일만 지원합니다.")
    return ext


def normalize_saved_ext(ext):
    """DECA TestData only accepts paths ending with jpg/png/bmp."""
    if ext == ".jpeg":
        return ".jpg"
    return ext


@app.route("/upload", methods=["POST"])
def upload():
    source_file = request.files.get("source_image")
    target_file = request.files.get("target_image")

    try:
        source_ext = validate_upload_file(source_file, "소스 이미지")
        target_ext = validate_upload_file(target_file, "타깃 이미지")
    except ValueError as e:
        return render_page(error=str(e))

    result_id = str(uuid.uuid4())[:8]
    work_dir = UPLOAD_DIR / result_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RESULT_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = work_dir / f"source{normalize_saved_ext(source_ext)}"
    target_path = work_dir / f"target{normalize_saved_ext(target_ext)}"
    source_file.save(str(source_path))
    target_file.save(str(target_path))

    try:
        t0 = time.time()
        print(f"\n[{result_id}] 처리 시작: source={source_file.filename}, target={target_file.filename}")

        print("  [1/4] Source DECA 복원...")
        try:
            source_result = deca_model.infer(str(source_path), out_dir=out_dir)
        except Exception as e:
            raise RuntimeError(f"소스 이미지에서 얼굴 복원 실패: {e}") from e

        print("  [2/4] Target 얼굴 기준 추출...")
        try:
            target_result = deca_model.extract_face_reference(str(target_path))
        except Exception as e:
            raise RuntimeError(f"타깃 이미지에서 얼굴 추출 실패: {e}") from e

        print("  [3/4] Source 3D 복원 결과 생성...")
        composite_rgb = create_composite(
            source_result["crop_rgb"],
            source_result["dense_2d"],
            source_result["dense_faces"],
            source_result["dense_colors"],
            source_result["lm2d"],
        )
        composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "source_reconstruction.png"), composite_bgr)

        source_bgr = cv2.imread(str(source_path))
        target_bgr = cv2.imread(str(target_path))
        if source_bgr is None or target_bgr is None:
            raise RuntimeError("업로드한 이미지를 다시 읽지 못했습니다.")
        cv2.imwrite(str(out_dir / "source_original.jpg"), source_bgr)
        cv2.imwrite(str(out_dir / "target_original.jpg"), target_bgr)

        print("  [4/4] Blender mask 렌더링 및 target 합성...")
        blender_render_and_composite(source_result, target_result, str(target_path), out_dir)

        elapsed = round(time.time() - t0, 1)
        stats = {
            "detail_verts": f"{source_result['n_face_verts']:,}",
            "detail_faces": f"{source_result['n_face_faces']:,}",
            "time": elapsed,
        }
        print(f"  [완료] {result_id} ({elapsed}s)")
        return render_page(result_id=result_id, stats=stats)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_page(error=f"처리 중 오류: {str(e)}")


@app.route("/image/<result_id>/<img_type>")
def serve_image(result_id, img_type):
    type_map = {
        "source_original": "source_original.jpg",
        "target_original": "target_original.jpg",
        "source_reconstruction": "source_reconstruction.png",
        "mask_render": "mask_render.png",
        "mask_composite": "mask_composite.png",
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
    allowed = {
        "detail.obj", "detail.ply", "coarse.ply", "full_head.ply",
        "source_reconstruction.png", "source_original.jpg", "target_original.jpg",
        "mask_render.png", "mask_composite.png",
    }
    if filename not in allowed:
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
