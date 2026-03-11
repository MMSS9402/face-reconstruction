"""
Blender PBR Mask Material Renderer
===================================
DECA face-only .obj에 실리콘/라텍스/레진 PBR 재질을 적용,
스튜디오 조명으로 사실적인 마스크 사진을 렌더링 (headless Cycles GPU).

Usage:
    blender --background --python render_blender.py -- \
        --obj detail.obj \
        --material all \
        --output_dir ./output \
        --samples 128 \
        --resolution 1024
"""

import bpy
import sys
import os
import argparse
import math
from pathlib import Path
from mathutils import Vector


# ── Material Presets ────────────────────────────────────────────────────

MATERIALS = {
    "silicone": {
        "subsurface_weight": 0.55,
        "subsurface_radius": (1.0, 0.4, 0.25),
        "subsurface_scale": 0.08,
        "roughness": 0.38,
        "specular_ior_level": 0.65,
        "coat_weight": 0.20,
        "coat_roughness": 0.15,
        "saturation_mult": 0.75,
        "brightness_mult": 1.0,
        "warm_tint": (1.03, 1.00, 0.92),
    },
    "latex": {
        "subsurface_weight": 0.12,
        "subsurface_radius": (0.4, 0.25, 0.15),
        "subsurface_scale": 0.03,
        "roughness": 0.60,
        "specular_ior_level": 0.35,
        "coat_weight": 0.08,
        "coat_roughness": 0.50,
        "saturation_mult": 0.85,
        "brightness_mult": 0.95,
        "warm_tint": (1.02, 1.01, 0.90),
    },
    "resin": {
        "subsurface_weight": 0.03,
        "subsurface_radius": (0.15, 0.08, 0.04),
        "subsurface_scale": 0.01,
        "roughness": 0.08,
        "specular_ior_level": 0.85,
        "coat_weight": 0.55,
        "coat_roughness": 0.05,
        "saturation_mult": 0.60,
        "brightness_mult": 1.05,
        "warm_tint": (1.00, 0.98, 0.88),
    },
}


# ── Scene Setup ─────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_obj(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    return obj


def setup_camera(obj, resolution=1024, cam_params=None):
    """Orthographic front-facing camera. After default OBJ import: Z-up, face toward -Y.
    If cam_params (scale, tx, ty) given, match DECA's orthographic projection for 224x224 crop."""
    bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    ys = [v.y for v in bbox]
    y_min = min(ys)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0

    if cam_params is not None:
        scale, tx, ty = cam_params
        cam_data.ortho_scale = 2.0 / scale
        cam_x = -tx
        cam_z = -ty
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = (cam_x, y_min - 5.0, cam_z)
        cam_obj.rotation_euler = (math.pi / 2, 0, 0)
        resolution = 224
    else:
        xs = [v.x for v in bbox]
        zs = [v.z for v in bbox]
        cx = (min(xs) + max(xs)) / 2
        cz = (min(zs) + max(zs)) / 2
        span = max(max(xs)-min(xs), max(zs)-min(zs))
        cam_data.ortho_scale = span * 1.15
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = (cx, y_min - 5.0, cz)
        cam_obj.rotation_euler = (math.pi / 2, 0, 0)

    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100

    return cam_obj


def setup_lighting(obj):
    """Bright warm studio lighting for realistic skin-like mask photography."""
    bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    y_min = min(ys)
    span = max(max(xs)-min(xs), max(zs)-min(zs))

    # Warm ambient environment
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 1.8
    bg.inputs["Color"].default_value = (1.0, 0.97, 0.92, 1.0)

    d = span * 5

    def add_light(name, energy, color, loc, size_mult=2.0):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy
        light.color = color
        light.size = span * size_mult
        light_obj = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = loc
        con = light_obj.constraints.new("TRACK_TO")
        tgt = bpy.data.objects.new(f"{name}_tgt", None)
        bpy.context.collection.objects.link(tgt)
        tgt.location = (cx, cy, cz)
        con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Z"

    # Key: strong warm light, front-right-above
    add_light("Key", 2.0, (1.0, 0.95, 0.88),
              (cx + d*0.4, y_min - d*0.9, cz + d*0.35), size_mult=4.0)

    # Fill: softer cool light, front-left
    add_light("Fill", 0.8, (0.95, 0.97, 1.0),
              (cx - d*0.5, y_min - d*0.7, cz + d*0.1), size_mult=5.0)

    # Top: overhead soft light
    add_light("Top", 0.5, (1.0, 0.98, 0.95),
              (cx, y_min - d*0.3, cz + d*0.8), size_mult=6.0)


def setup_render(samples=128):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type != "CPU"

    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True

    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    scene.view_settings.view_transform = "Standard"


# ── Material Creation ──────────────────────────────────────────────────

def get_vertex_color_attr(obj):
    mesh = obj.data
    if mesh.color_attributes:
        return mesh.color_attributes[0].name
    if mesh.vertex_colors:
        return mesh.vertex_colors[0].name
    return None


def create_mask_material(obj, preset_name):
    preset = MATERIALS[preset_name]
    mat = bpy.data.materials.new(name=f"Mask_{preset_name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (800, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    vc_attr = get_vertex_color_attr(obj)
    if vc_attr:
        vc_node = nodes.new("ShaderNodeVertexColor")
        vc_node.layer_name = vc_attr
        vc_node.location = (-800, 0)

        # sRGB gamma correction: vertex colors are stored linear in Blender,
        # apply gamma to restore natural brightness
        gamma = nodes.new("ShaderNodeGamma")
        gamma.location = (-600, 0)
        gamma.inputs["Gamma"].default_value = 1.8
        links.new(vc_node.outputs["Color"], gamma.inputs["Color"])

        # Warm skin-tone tint per material
        tint_r, tint_g, tint_b = preset["warm_tint"]
        mix_rgb = nodes.new("ShaderNodeMix")
        mix_rgb.data_type = 'RGBA'
        mix_rgb.location = (-400, 0)
        mix_rgb.inputs["Factor"].default_value = 1.0
        mix_rgb.blend_type = 'MULTIPLY'
        mix_rgb.inputs[7].default_value = (tint_r, tint_g, tint_b, 1.0)  # B_Color
        links.new(gamma.outputs["Color"], mix_rgb.inputs[6])  # A_Color

        # Saturation / brightness adjustment
        hsv = nodes.new("ShaderNodeHueSaturation")
        hsv.location = (-200, 0)
        hsv.inputs["Hue"].default_value = 0.5
        hsv.inputs["Saturation"].default_value = preset["saturation_mult"]
        hsv.inputs["Value"].default_value = preset["brightness_mult"]
        links.new(mix_rgb.outputs[2], hsv.inputs["Color"])  # Result → HSV

        links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.55, 0.45, 1.0)

    # PBR parameters
    bsdf.inputs["Subsurface Weight"].default_value = preset["subsurface_weight"]
    bsdf.inputs["Subsurface Radius"].default_value = preset["subsurface_radius"]
    bsdf.inputs["Subsurface Scale"].default_value = preset["subsurface_scale"]
    bsdf.inputs["Roughness"].default_value = preset["roughness"]
    bsdf.inputs["Specular IOR Level"].default_value = preset["specular_ior_level"]
    bsdf.inputs["Coat Weight"].default_value = preset["coat_weight"]
    bsdf.inputs["Coat Roughness"].default_value = preset["coat_roughness"]

    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ── Main ───────────────────────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender PBR mask renderer")
    parser.add_argument("--obj", required=True, help="Path to .obj file")
    parser.add_argument("--material", default="all",
                        choices=["silicone", "latex", "resin", "all"])
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as obj)")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--cam_params", type=float, nargs=3, default=None,
                        metavar=("SCALE", "TX", "TY"),
                        help="DECA camera params for 224x224 crop-space matching")
    return parser.parse_args(argv)


def render_material(obj, material_name, output_path, samples, resolution):
    print(f"\n{'='*60}")
    print(f"  Rendering: {material_name}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    create_mask_material(obj, material_name)
    setup_render(samples=samples)

    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print(f"  Done: {output_path}")


def main():
    args = parse_args()
    obj_path = os.path.abspath(args.obj)
    output_dir = Path(args.output_dir) if args.output_dir else Path(obj_path).parent

    materials = list(MATERIALS.keys()) if args.material == "all" else [args.material]

    clear_scene()
    obj = import_obj(obj_path)
    setup_camera(obj, resolution=args.resolution, cam_params=args.cam_params)
    setup_lighting(obj)

    for mat_name in materials:
        out_path = output_dir / f"{mat_name}_render.png"
        render_material(obj, mat_name, out_path, args.samples, args.resolution)

    print(f"\nAll renders complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
