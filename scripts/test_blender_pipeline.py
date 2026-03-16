"""
End-to-end test: DECA inference -> single mask render -> composite.

Usage:
    conda run -n deca-env python scripts/test_blender_pipeline.py \
        --image /path/to/face.jpg \
        --output_dir outputs/blender_test
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

BLENDER = str(Path(__file__).resolve().parent.parent.parent / "tools" / "blender-4.2.13-linux-x64" / "blender")
RENDER_SCRIPT = str(Path(__file__).resolve().parent / "render_blender.py")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "synthetic_3d_attack" / "third_party" / "DECA"))


def run_deca(image_path, out_dir):
    from webapp_deca import DECAModel

    print("\n[1/3] DECA Inference...")
    t0 = time.time()
    model = DECAModel(device="cuda")
    result = model.infer(str(image_path), out_dir=str(out_dir))
    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    print(f"  Face vertices: {result['n_face_verts']:,}")
    print(f"  Face faces: {result['n_face_faces']:,}")

    np.save(str(out_dir / "tform_params.npy"), result["tform_params"])
    np.save(str(out_dir / "lm2d.npy"), result["lm2d"])
    np.save(str(out_dir / "cam_params.npy"), result["cam_params"])

    original_bgr = cv2.imread(str(image_path))
    cv2.imwrite(str(out_dir / "original.jpg"), original_bgr)
    return result


def run_blender(obj_path, out_dir, cam_params, samples=64):
    print("\n[2/3] Blender single mask render...")
    t0 = time.time()

    cmd = [
        BLENDER, "--background", "--python", RENDER_SCRIPT, "--",
        "--obj", str(obj_path),
        "--output_dir", str(out_dir),
        "--samples", str(samples),
        "--cam_params", str(cam_params[0]), str(cam_params[1]), str(cam_params[2]),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  Blender ERROR:\n{(proc.stderr or proc.stdout)[-500:]}")
        return False

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    return True


def run_composite(out_dir):
    from composite_blender import composite_render

    print("\n[3/3] Compositing onto original...")
    t0 = time.time()

    original_bgr = cv2.imread(str(out_dir / "original.jpg"))
    tform_params = np.load(str(out_dir / "tform_params.npy"))
    lm2d = np.load(str(out_dir / "lm2d.npy"))
    render_rgba = cv2.imread(str(out_dir / "mask_render.png"), cv2.IMREAD_UNCHANGED)

    result = composite_render(
        original_bgr,
        render_rgba,
        tform_params,
        lm2d,
        source_lm2d=lm2d,
    )
    cv2.imwrite(str(out_dir / "mask_composite.png"), result)

    h, w = original_bgr.shape[:2]
    target_h = 512
    target_w = int(w * (target_h / h))
    original_resized = cv2.resize(original_bgr, (target_w, target_h))
    composite_resized = cv2.resize(result, (target_w, target_h))
    grid = np.hstack([original_resized, composite_resized])

    cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(grid, "Mask Composite", (target_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir / "comparison_grid.png"), grid)

    elapsed = time.time() - t0
    print(f"  Composite done ({elapsed:.1f}s)")
    print(f"  Output: {out_dir / 'comparison_grid.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default="outputs/blender_test")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Blender Mask Pipeline Test")
    print("=" * 60)

    result = run_deca(args.image, out_dir)
    obj_path = out_dir / "detail.obj"
    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found")
        return

    ok = run_blender(obj_path, out_dir, result["cam_params"], samples=args.samples)
    if not ok:
        return

    run_composite(out_dir)

    print("\n" + "=" * 60)
    print("  Complete!")
    print(f"  Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
