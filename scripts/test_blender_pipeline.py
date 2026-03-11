"""
End-to-end test: DECA inference → Blender PBR render → Composite.
Produces: original | silicone | latex | resin comparison grid.

Usage:
    conda run -n deca-env python scripts/test_blender_pipeline.py \
        --image /path/to/face.jpg \
        --output_dir outputs/blender_test
"""

import sys
import os
import argparse
import subprocess
import time
import json
from pathlib import Path

import cv2
import numpy as np

BLENDER = str(Path(__file__).resolve().parent.parent.parent / "tools" / "blender-4.2.13-linux-x64" / "blender")
RENDER_SCRIPT = str(Path(__file__).resolve().parent / "render_blender.py")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "synthetic_3d_attack" / "third_party" / "DECA"))


def run_deca(image_path, out_dir):
    """Run DECA inference and save necessary artifacts."""
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

    orig_bgr = cv2.imread(str(image_path))
    cv2.imwrite(str(out_dir / "original.jpg"), orig_bgr)

    return result


def run_blender(obj_path, out_dir, cam_params, samples=64):
    """Run Blender headless rendering for all 3 materials."""
    print("\n[2/3] Blender PBR Rendering (silicone / latex / resin)...")
    t0 = time.time()

    cmd = [
        BLENDER, "--background", "--python", RENDER_SCRIPT, "--",
        "--obj", str(obj_path),
        "--material", "all",
        "--output_dir", str(out_dir),
        "--samples", str(samples),
        "--cam_params", str(cam_params[0]), str(cam_params[1]), str(cam_params[2]),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  Blender ERROR:\n{proc.stderr[-500:]}")
        return False

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    return True


def run_composite(out_dir):
    """Composite Blender renders onto original image."""
    from composite_blender import composite_render, MATERIAL_NAMES

    print("\n[3/3] Compositing onto original...")
    t0 = time.time()

    original_bgr = cv2.imread(str(out_dir / "original.jpg"))
    tform_params = np.load(str(out_dir / "tform_params.npy"))
    lm2d = np.load(str(out_dir / "lm2d.npy"))

    composites = {}
    for mat_name in MATERIAL_NAMES:
        render_path = out_dir / f"{mat_name}_render.png"
        if not render_path.exists():
            print(f"  [SKIP] {mat_name}")
            continue

        render_rgba = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
        result = composite_render(original_bgr, render_rgba, tform_params, lm2d)
        out_path = out_dir / f"{mat_name}_composite.png"
        cv2.imwrite(str(out_path), result)
        composites[mat_name] = result
        print(f"  [OK] {mat_name}")

    # Comparison grid: original | silicone | latex | resin
    h, w = original_bgr.shape[:2]
    target_h = 512
    scale = target_h / h

    panels = [original_bgr]
    labels = ["Original"]
    for name in MATERIAL_NAMES:
        if name in composites:
            panels.append(composites[name])
            labels.append(name.capitalize())

    resized = []
    for p in panels:
        rp = cv2.resize(p, (int(w * scale), target_h))
        resized.append(rp)

    grid = np.hstack(resized)

    for i, label in enumerate(labels):
        x = i * resized[0].shape[1] + 10
        cv2.putText(grid, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(grid, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    cv2.imwrite(str(out_dir / "comparison_grid.png"), grid)
    print(f"\n  Comparison grid: {out_dir / 'comparison_grid.png'}")

    elapsed = time.time() - t0
    print(f"  Composite done ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default="outputs/blender_test")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Blender PBR Mask Pipeline Test")
    print("=" * 60)

    result = run_deca(args.image, out_dir)

    cam_params = result["cam_params"]
    obj_path = out_dir / "detail.obj"
    if not obj_path.exists():
        print(f"ERROR: {obj_path} not found")
        return

    ok = run_blender(obj_path, out_dir, cam_params, samples=args.samples)
    if not ok:
        return

    run_composite(out_dir)

    print("\n" + "=" * 60)
    print("  Complete!")
    print(f"  Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
