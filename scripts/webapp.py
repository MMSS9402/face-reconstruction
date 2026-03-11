"""
Flask 웹앱: 사진 업로드 → Deep3DFaceRecon 3D 복원 + 웹 3D 뷰어

사용법:
    conda run -n deep3d python scripts/webapp.py
    → http://localhost:5000
"""

import sys
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.io import loadmat, savemat
from flask import Flask, request, render_template_string, send_file

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DEEP3D = ROOT / "methods" / "Deep3DFaceRecon_pytorch"
BFM_DIR = DEEP3D / "BFM"
UPLOAD_DIR = ROOT / "webapp_uploads"
RESULT_DIR = ROOT / "webapp_results"

FOCAL = 1015.0
CENTER = 112.0
CAMERA_D = 10.0
CROP_SIZE = 224

sys.path.insert(0, str(DEEP3D))


# ===========================================================================
# Deep3DFaceRecon Model
# ===========================================================================

class Deep3DModel:
    def __init__(self):
        from argparse import Namespace
        from models import create_model
        from util.load_mats import load_lm3d
        from util.preprocess import align_img

        self.align_img = align_img
        self.opt = Namespace(
            name="face_recon_feat0.2_augment", model="facerecon", epoch="20",
            checkpoints_dir=str(DEEP3D / "checkpoints"),
            bfm_folder=str(BFM_DIR), bfm_model="BFM_model_front.mat",
            net_recon="resnet50", use_last_fc=False,
            init_path=str(DEEP3D / "checkpoints" / "init_model" / "resnet50-0676ba61.pth"),
            isTrain=False, phase="test", dataset_mode=None, img_folder="", suffix="",
            gpu_ids="0", ddp_port="12355", use_ddp=False, use_opengl=False,
            verbose=False, world_size=1, display_per_batch=True,
            eval_batch_nums=float("inf"), vis_batch_nums=1, add_image=True,
            focal=FOCAL, center=CENTER, camera_d=CAMERA_D, z_near=5.0, z_far=15.0,
        )

        print("  Deep3DFaceRecon 모델 로딩...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.device = self.device
        self.model.parallelize()
        self.model.eval()
        self.lm3d_std = load_lm3d(self.opt.bfm_folder)
        print("  Deep3DFaceRecon 로딩 완료!")

    def infer(self, img_path, lm5, out_dir=None):
        """Run inference.
        Returns (coeffs_dict, vertex_colors (N,3) float [0,1], crop_rgb (224,224,3) uint8).
        If out_dir is set, saves .obj, .ply, .mat there.
        """
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        lm = lm5.copy().astype(np.float32)
        lm[:, -1] = H - 1 - lm[:, -1]

        _, im_aligned, lm_aligned, _ = self.align_img(im, lm, self.lm3d_std)

        im_tensor = torch.tensor(
            np.array(im_aligned) / 255.0, dtype=torch.float32
        ).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            im_on_device = im_tensor.to(self.device)
            output_coeff = self.model.net_recon(im_on_device)
            self.model.facemodel.to(self.device)
            pred_vertex, pred_tex, pred_color, pred_lm = \
                self.model.facemodel.compute_for_render(output_coeff)
            pred_coeffs = self.model.facemodel.split_coeff(output_coeff)

        coeffs = {k: v.cpu().numpy() for k, v in pred_coeffs.items()}
        vertex_colors = pred_color.cpu().numpy()[0]
        crop_rgb = np.array(im_aligned)

        pred_lm_np = pred_lm.cpu().numpy()[0]
        pred_lm_np[:, 1] = CROP_SIZE - 1 - pred_lm_np[:, 1]
        coeffs["lm68"] = pred_lm_np.reshape(1, 68, 2)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            savemat(str(out_dir / "coeffs.mat"), coeffs)

            recon_shape = pred_vertex.cpu().numpy()[0].copy()
            recon_shape[..., -1] = CAMERA_D - recon_shape[..., -1]
            tri = self.model.facemodel.face_buf.cpu().numpy()
            mesh = trimesh.Trimesh(
                vertices=recon_shape, faces=tri,
                vertex_colors=np.clip(255.0 * vertex_colors, 0, 255).astype(np.uint8),
                process=False,
            )
            mesh.export(str(out_dir / "mesh.obj"))
            mesh.export(str(out_dir / "mesh.ply"))

        return coeffs, vertex_colors, crop_rgb


# ===========================================================================
# MTCNN Landmark Detection
# ===========================================================================

def detect_landmarks(img_path):
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(select_largest=True, device="cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(img_path).convert("RGB")
    _, _, landmarks = mtcnn.detect(img, landmarks=True)
    if landmarks is None:
        return None
    return landmarks[0]


# ===========================================================================
# BFM Model (geometry reconstruction)
# ===========================================================================

class BFMModel:
    def __init__(self, bfm_folder):
        model = loadmat(str(bfm_folder / "BFM_model_front.mat"))
        self.mean_shape = model["meanshape"].astype(np.float32)
        self.id_base = model["idBase"].astype(np.float32)
        self.exp_base = model["exBase"].astype(np.float32)
        self.face_buf = model["tri"].astype(np.int64) - 1
        self.point_buf = model["point_buf"].astype(np.int64) - 1

        ms = self.mean_shape.reshape(-1, 3)
        ms -= np.mean(ms, axis=0, keepdims=True)
        self.mean_shape = ms.reshape(-1, 1)

        self.persc_proj = np.array([
            FOCAL, 0, CENTER, 0, FOCAL, CENTER, 0, 0, 1,
        ], dtype=np.float32).reshape(3, 3).T

    def compute_shape(self, id_c, exp_c):
        return (self.id_base @ id_c + self.exp_base @ exp_c + self.mean_shape.flatten()).reshape(-1, 3)

    def compute_rotation(self, angles):
        x, y, z = angles
        Rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        return (Rz @ Ry @ Rx).T

    def compute_normals(self, vertices):
        f = self.face_buf
        v0, v1, v2 = vertices[f[:,0]], vertices[f[:,1]], vertices[f[:,2]]
        fn = np.cross(v0 - v1, v1 - v2)
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
        fn = np.vstack([fn, np.zeros((1, 3))])
        vn = np.sum(fn[self.point_buf], axis=1)
        return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)

    def reconstruct_geometry(self, coeffs):
        id_c = coeffs["id"].flatten()
        exp_c = coeffs["exp"].flatten()
        angles = coeffs["angle"].flatten()
        trans = coeffs["trans"].flatten()
        lm68 = coeffs["lm68"][0]

        shape = self.compute_shape(id_c, exp_c)
        rot = self.compute_rotation(angles)
        shape_t = shape @ rot + trans
        shape_cam = shape_t.copy()
        shape_cam[:, 2] = CAMERA_D - shape_cam[:, 2]

        proj = shape_cam @ self.persc_proj
        verts_2d = proj[:, :2] / proj[:, 2:3]
        verts_2d[:, 1] = CROP_SIZE - 1 - verts_2d[:, 1]

        normals_rot = self.compute_normals(shape) @ rot
        return verts_2d, normals_rot, shape_cam[:, 2], lm68


# ===========================================================================
# Rasterizer (normals + depth + mask + vertex color)
# ===========================================================================

def rasterize_geometry(verts_2d, normals, depths, faces,
                       vertex_colors=None, img_size=CROP_SIZE):
    """Rasterize mesh with z-buffer. Optionally interpolates vertex colors."""
    normal_map = np.zeros((img_size, img_size, 3), dtype=np.float32)
    depth_map = np.full((img_size, img_size), np.inf, dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    has_color = vertex_colors is not None
    color_map = np.zeros((img_size, img_size, 3), dtype=np.float32) if has_color else None

    for f in faces:
        v0, v1, v2 = verts_2d[f[0]], verts_2d[f[1]], verts_2d[f[2]]
        n0, n1, n2 = normals[f[0]], normals[f[1]], normals[f[2]]
        z0, z1, z2 = depths[f[0]], depths[f[1]], depths[f[2]]
        if has_color:
            c0, c1, c2 = vertex_colors[f[0]], vertex_colors[f[1]], vertex_colors[f[2]]

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
                if has_color:
                    color_map[py, px] = w0i[i]*c0 + w1i[i]*c1 + w2i[i]*c2

    depth_map[mask == 0] = 0.0
    return normal_map, depth_map, mask, color_map


def create_composite(crop_rgb, color_map, mask, lm68):
    """Create 3-panel composite like outputs/deep3d/*.png:
    [input_crop | 3D rendered overlay | 68 landmarks]
    """
    h, w = crop_rgb.shape[:2]
    crop_f = crop_rgb.astype(np.float32)
    rendered_f = np.clip(color_map, 0, 1) * 255.0
    m3 = mask[:, :, np.newaxis]

    panel1 = crop_rgb.copy()
    panel2 = (crop_f * (1 - m3) + rendered_f * m3).astype(np.uint8)

    panel3 = panel2.copy()
    for pt in lm68:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(panel3, (x, y), 2, (255, 0, 0), -1)

    return np.hstack([panel1, panel2, panel3])


# ===========================================================================
# HTML Template
# ===========================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep3D Face Reconstruction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 32px;
            text-align: center;
            border-bottom: 1px solid #333;
        }
        .header h1 { font-size: 28px; color: #fff; margin-bottom: 8px; }
        .header p { color: #888; font-size: 14px; }
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
        .upload-section:hover { border-color: #4a9eff; }
        .upload-section label {
            display: inline-block;
            background: #4a9eff;
            color: #fff;
            padding: 14px 32px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background 0.3s;
        }
        .upload-section label:hover { background: #3a8eef; }
        .upload-section input[type="file"] { display: none; }
        .upload-section .hint { margin-top: 12px; color: #666; font-size: 13px; }

        .processing { text-align: center; padding: 60px; display: none; }
        .spinner {
            width: 48px; height: 48px;
            border: 4px solid #333;
            border-top: 4px solid #4a9eff;
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
            transition: background 0.2s;
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
        .composite-labels span {
            flex: 1;
            color: #888;
            font-size: 13px;
        }

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

        .pipeline-info {
            background: #111;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin-top: 24px;
        }
        .pipeline-info h3 { color: #4a9eff; margin-bottom: 12px; }
        .pipeline-info ol { padding-left: 20px; }
        .pipeline-info li { margin-bottom: 6px; color: #999; font-size: 13px; }
        .pipeline-info li span { color: #ccc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Deep3D Face Reconstruction</h1>
        <p>사진 업로드 → 3D 얼굴 복원 + 인터랙티브 3D 뷰어</p>
    </div>
    <div class="container">
        <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
            <div class="upload-section" id="dropZone">
                <label for="fileInput">얼굴 사진 업로드</label>
                <input type="file" id="fileInput" name="image" accept="image/jpeg,image/png"
                       onchange="submitForm()">
                <p class="hint">JPG / PNG 형식. 정면 얼굴이 포함된 이미지를 업로드하세요.</p>
            </div>
        </form>

        <div class="processing" id="processing">
            <div class="spinner"></div>
            <p>3D 복원 중... (약 30초~2분 소요)</p>
            <p style="color:#666; font-size:13px; margin-top:8px;">
                랜드마크 검출 → 3D 복원 → 메시 래스터라이즈 → 합성 이미지 생성
            </p>
        </div>

        {% if result_id %}
        <!-- 3D Viewer -->
        <div class="section">
            <h2>3D 메시 뷰어</h2>
            <div id="viewer3d"></div>
            <div class="viewer-controls">
                <button onclick="resetCamera()">카메라 리셋</button>
                <button onclick="toggleWireframe()">와이어프레임 토글</button>
                <button onclick="toggleLight()">조명 토글</button>
            </div>
        </div>

        <!-- Composite Image -->
        <div class="section">
            <h2>3D 복원 결과</h2>
            <img class="composite-img" src="/image/{{ result_id }}/composite" alt="Composite">
            <div class="composite-labels">
                <span>Input Crop</span>
                <span>3D Reconstruction</span>
                <span>68 Landmarks</span>
            </div>
        </div>

        <!-- Downloads -->
        <div class="section">
            <h2>다운로드</h2>
            <div class="actions">
                <a class="download-btn" style="background:#4a6aff;"
                   href="/file/{{ result_id }}/mesh.obj">3D 메시 (.obj)</a>
                <a class="download-btn" style="background:#6a4aff;"
                   href="/file/{{ result_id }}/mesh.ply">3D 메시 (.ply)</a>
                <a class="download-btn" style="background:#ff8c4a;"
                   href="/file/{{ result_id }}/coeffs.mat">BFM 계수 (.mat)</a>
                <a class="download-btn" style="background:#2d8f4e;"
                   href="/image/{{ result_id }}/composite?download=1">합성 이미지 (.png)</a>
                <a class="download-btn" style="background:#555;"
                   href="/image/{{ result_id }}/original?download=1">원본 이미지</a>
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
            <h3>파이프라인 설명</h3>
            <ol>
                <li><span>MTCNN 랜드마크 검출</span> — 얼굴에서 5-point landmark 추출</li>
                <li><span>Deep3DFaceRecon Inference</span> — BFM 파라미터 회귀 (shape, expression, pose, lighting)</li>
                <li><span>3D 메시 생성</span> — .obj / .ply 형식으로 vertex color 포함 메시 저장</li>
                <li><span>래스터라이즈</span> — numpy 래스터라이저로 2D 렌더링 (vertex color + 랜드마크)</li>
                <li><span>합성 이미지</span> — Input Crop | 3D Overlay | 68 Landmarks</li>
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
        45, container.clientWidth / container.clientHeight, 0.1, 1000
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

    let meshObj = null;
    let wireframeOn = false;
    let lightOn = true;

    const loader = new PLYLoader();
    loader.load('/mesh/{{ result_id }}', function(geometry) {
        geometry.computeVertexNormals();
        const material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.5,
            metalness: 0.0,
        });
        meshObj = new THREE.Mesh(geometry, material);

        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        meshObj.position.sub(center);
        scene.add(meshObj);

        const size = new THREE.Vector3();
        geometry.boundingBox.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        camera.position.set(0, 0, maxDim * 2.5);
        controls.target.set(0, 0, 0);
        controls.update();
    });

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    window.resetCamera = function() {
        if (meshObj) {
            const box = new THREE.Box3().setFromObject(meshObj);
            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            camera.position.set(0, 0, maxDim * 2.5);
            controls.target.set(0, 0, 0);
            controls.update();
        }
    };

    window.toggleWireframe = function() {
        if (meshObj) {
            wireframeOn = !wireframeOn;
            meshObj.material.wireframe = wireframeOn;
        }
    };

    window.toggleLight = function() {
        lightOn = !lightOn;
        dirLight.visible = lightOn;
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
deep3d_model = None
bfm = None


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, result_id=None, error=None)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return render_template_string(HTML_TEMPLATE, result_id=None,
                                      error="파일이 선택되지 않았습니다.")
    file = request.files["image"]
    if file.filename == "":
        return render_template_string(HTML_TEMPLATE, result_id=None,
                                      error="파일이 선택되지 않았습니다.")

    result_id = str(uuid.uuid4())[:8]
    work_dir = UPLOAD_DIR / result_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RESULT_DIR / result_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        return render_template_string(HTML_TEMPLATE, result_id=None,
                                      error="JPG 또는 PNG 파일만 지원합니다.")

    img_path = work_dir / f"input{ext}"
    file.save(str(img_path))

    try:
        print(f"\n[{result_id}] 처리 시작: {file.filename}")

        # 1. Landmark detection
        print(f"  [1/3] 랜드마크 검출...")
        lm5 = detect_landmarks(str(img_path))
        if lm5 is None:
            return render_template_string(HTML_TEMPLATE, result_id=None,
                error="얼굴을 검출할 수 없습니다. 정면 얼굴이 잘 보이는 이미지를 사용해주세요.")

        # 2. Deep3D inference → .obj/.ply/.mat saved to out_dir
        print(f"  [2/3] Deep3DFaceRecon inference...")
        coeffs, vertex_colors, crop_rgb = deep3d_model.infer(
            str(img_path), lm5, out_dir=out_dir
        )

        # 3. Rasterize mesh with vertex colors → composite image
        print(f"  [3/3] 래스터라이즈 + 합성 이미지 생성...")
        verts_2d, normals_rot, depths, lm68 = bfm.reconstruct_geometry(coeffs)
        normal_crop, depth_crop, mask_crop, color_crop = rasterize_geometry(
            verts_2d, normals_rot, depths, bfm.face_buf,
            vertex_colors=vertex_colors,
        )

        composite_rgb = create_composite(crop_rgb, color_crop, mask_crop, lm68)
        composite_bgr = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "composite.png"), composite_bgr)

        orig_bgr = cv2.imread(str(img_path))
        cv2.imwrite(str(out_dir / "original.jpg"), orig_bgr)

        print(f"  [완료] {result_id}")
        return render_template_string(HTML_TEMPLATE, result_id=result_id, error=None)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template_string(HTML_TEMPLATE, result_id=None,
                                      error=f"처리 중 오류 발생: {str(e)}")


@app.route("/image/<result_id>/<img_type>")
def serve_image(result_id, img_type):
    type_map = {"original": "original.jpg", "composite": "composite.png"}
    if img_type not in type_map:
        return "Not found", 404
    path = RESULT_DIR / result_id / type_map[img_type]
    if not path.exists():
        return "Not found", 404
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    if request.args.get("download"):
        return send_file(str(path), as_attachment=True,
                         download_name=f"{img_type}_{result_id}{path.suffix}")
    return send_file(str(path), mimetype=mime)


@app.route("/mesh/<result_id>")
def serve_mesh(result_id):
    """Serve .ply for Three.js viewer."""
    path = RESULT_DIR / result_id / "mesh.ply"
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), mimetype="application/octet-stream")


@app.route("/file/<result_id>/<filename>")
def serve_file(result_id, filename):
    ALLOWED = {"mesh.obj", "mesh.ply", "coeffs.mat"}
    if filename not in ALLOWED:
        return "Not found", 404
    path = RESULT_DIR / result_id / filename
    if not path.exists():
        return "Not found", 404
    return send_file(str(path), as_attachment=True,
                     download_name=f"{result_id}_{filename}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Deep3D Face Reconstruction 초기화 중...")
    print("=" * 60)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    os.chdir(str(DEEP3D))

    print("\n[1/2] Deep3DFaceRecon 모델 로딩...")
    deep3d_model = Deep3DModel()

    print("\n[2/2] BFM 모델 로딩...")
    bfm = BFMModel(BFM_DIR)
    print(f"  vertices: {bfm.mean_shape.size // 3}, faces: {len(bfm.face_buf)}")

    print("\n" + "=" * 60)
    print("  서버 시작: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
