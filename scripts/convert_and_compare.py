"""
Deep3DFaceRecon과 EMOCA의 .obj 결과물을 .ply로 변환하고,
두 방법의 결과를 비교하는 이미지 그리드를 생성한다.

사용법:
    python convert_and_compare.py
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent


def obj_with_vertex_colors_to_ply(obj_path: Path, ply_path: Path):
    """vertex color가 포함된 .obj를 .ply로 변환"""
    vertices, colors, faces = [], [], []

    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 7:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                if max(r, g, b) <= 1.0:
                    colors.append([int(r * 255), int(g * 255), int(b * 255)])
                else:
                    colors.append([int(r), int(g), int(b)])
            elif parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([200, 200, 200])
            elif parts[0] == "f":
                face_verts = []
                for p in parts[1:]:
                    face_verts.append(int(p.split("/")[0]) - 1)
                faces.append(face_verts)

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64) if faces else None
    colors_arr = np.array(colors, dtype=np.uint8)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors_arr,
        process=False,
    )
    mesh.export(str(ply_path))
    return mesh


def convert_deep3d(output_dir: Path, ply_dir: Path):
    """Deep3DFaceRecon .obj -> .ply 변환"""
    obj_files = sorted(output_dir.glob("*.obj"))
    results = {}
    for obj_path in obj_files:
        name = obj_path.stem
        ply_path = ply_dir / f"deep3d_{name}.ply"
        obj_with_vertex_colors_to_ply(obj_path, ply_path)
        print(f"  [Deep3D] {obj_path.name} -> {ply_path.name}")
        results[name] = ply_path
    return results


def convert_emoca(output_dir: Path, ply_dir: Path):
    """EMOCA .obj -> .ply 변환 (detail mesh 우선)"""
    results = {}
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        detail_obj = subdir / "mesh_coarse_detail.obj"
        coarse_obj = subdir / "mesh_coarse.obj"
        obj_path = detail_obj if detail_obj.exists() else coarse_obj
        if not obj_path.exists():
            continue

        name = subdir.name
        ply_path = ply_dir / f"emoca_{name}.ply"
        obj_with_vertex_colors_to_ply(obj_path, ply_path)
        print(f"  [EMOCA]  {obj_path.name} ({name}) -> {ply_path.name}")
        results[name] = ply_path
    return results


def make_comparison_grid(test_images_dir: Path, deep3d_out: Path, emoca_out: Path, save_path: Path):
    """원본 / Deep3D geometry / EMOCA geometry 비교 그리드"""
    test_images = sorted(test_images_dir.glob("*.jpg"))
    if not test_images:
        test_images = sorted(test_images_dir.glob("*.png"))

    deep3d_imgs = {}
    for p in sorted(deep3d_out.glob("*.png")):
        deep3d_imgs[p.stem] = p

    emoca_imgs = {}
    for subdir in sorted(emoca_out.iterdir()):
        if not subdir.is_dir():
            continue
        geom = subdir / "geometry_detail.png"
        if not geom.exists():
            geom = subdir / "geometry_coarse.png"
        if geom.exists():
            emoca_imgs[subdir.name] = geom

    n = len(test_images)
    cell_w, cell_h = 256, 256
    cols = 3
    grid = Image.new("RGB", (cell_w * cols, cell_h * n), (255, 255, 255))

    for i, img_path in enumerate(test_images):
        orig = Image.open(img_path).convert("RGB").resize((cell_w, cell_h))
        grid.paste(orig, (0, i * cell_h))

        stem = img_path.stem
        if stem in deep3d_imgs:
            d3d = Image.open(deep3d_imgs[stem]).convert("RGB").resize((cell_w, cell_h))
            grid.paste(d3d, (cell_w, i * cell_h))

        emoca_key = stem.replace("celeba_hq_", "celeba_hq_000") + "0"
        for key in emoca_imgs:
            if stem.replace("celeba_hq_", "") in key:
                emoca_key = key
                break
        if emoca_key in emoca_imgs:
            emo = Image.open(emoca_imgs[emoca_key]).convert("RGB").resize((cell_w, cell_h))
            grid.paste(emo, (cell_w * 2, i * cell_h))

    grid.save(str(save_path))
    print(f"\n비교 그리드 저장: {save_path}")
    print(f"  열: 원본 / Deep3DFaceRecon / EMOCA")
    print(f"  행: {n}개 이미지")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep3d_dir", default=str(ROOT / "outputs" / "deep3d"))
    parser.add_argument("--emoca_dir", default=str(ROOT / "outputs" / "emoca" / "EMOCA_v2_lr_mse_20"))
    parser.add_argument("--test_images", default=str(ROOT / "data" / "test_images"))
    parser.add_argument("--ply_dir", default=str(ROOT / "outputs" / "ply"))
    parser.add_argument("--grid_path", default=str(ROOT / "outputs" / "comparison_grid.png"))
    args = parser.parse_args()

    deep3d_dir = Path(args.deep3d_dir)
    emoca_dir = Path(args.emoca_dir)
    test_images = Path(args.test_images)
    ply_dir = Path(args.ply_dir)
    ply_dir.mkdir(parents=True, exist_ok=True)

    print("=== .obj -> .ply 변환 ===")
    convert_deep3d(deep3d_dir, ply_dir)
    convert_emoca(emoca_dir, ply_dir)

    print("\n=== 비교 그리드 생성 ===")
    make_comparison_grid(test_images, deep3d_dir, emoca_dir, Path(args.grid_path))

    print(f"\n=== PLY 파일 목록 (MeshLab으로 열기) ===")
    for p in sorted(ply_dir.glob("*.ply")):
        print(f"  meshlab {p}")


if __name__ == "__main__":
    main()
