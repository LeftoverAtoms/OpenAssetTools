bl_info = {
    "name": "OpenAssetTools Material Importer",
    "author": "OpenAssetTools",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "File > Import > OpenAssetTools Materials",
    "description": "Populate Blender materials from OpenAssetTools material JSON files and matching image files.",
    "category": "Import-Export",
}

import json
import os
import re
from pathlib import Path

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".tif",
    ".tiff",
    ".bmp",
    ".dds",
    ".exr",
    ".hdr",
    ".webp",
}

def normalize_name(name):
    return name.casefold().replace("\\", "/")


def sanitize_name(name):
    return re.sub(r"[^a-z0-9]+", "", normalize_name(name))


def without_known_texture_suffix(name):
    value = normalize_name(name)
    return re.sub(r"(_c|_n|_s|_o|_r|_g|_col|_color|_normal|_spec|_specular|_gloss|_occ|_ao)$", "", value)


def image_lookup_keys(name):
    if not name:
        return []

    value = normalize_name(str(name))
    path = Path(value)
    stem = normalize_name(path.stem)
    filename = normalize_name(path.name)
    slashless = value.replace("/", "")
    suffixless = without_known_texture_suffix(stem)

    keys = {
        value,
        filename,
        stem,
        slashless,
        slashless.replace(".", ""),
        suffixless,
        sanitize_name(value),
        sanitize_name(filename),
        sanitize_name(stem),
        sanitize_name(slashless),
        sanitize_name(suffixless),
    }

    if "/" in value:
        tail = value.split("/")[-1]
        keys.add(tail)
        keys.add(Path(tail).stem)
        keys.add(sanitize_name(tail))
        keys.add(sanitize_name(Path(tail).stem))

    return [key for key in keys if key]


def strip_blender_suffix(name):
    return re.sub(r"\.\d{3}$", "", name)


def index_images(root):
    images = {}

    if not root or not root.is_dir():
        return images

    for directory, _, filenames in os.walk(root):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue

            relative_without_extension = path.relative_to(root).with_suffix("").as_posix()
            for key in image_lookup_keys(path.stem):
                images.setdefault(key, path)
            for key in image_lookup_keys(path.name):
                images.setdefault(key, path)
            for key in image_lookup_keys(relative_without_extension):
                images.setdefault(key, path)
            for key in image_lookup_keys(relative_without_extension.split("/", 1)[-1]):
                images.setdefault(key, path)

    return images


def count_image_files(root):
    if not root or not root.is_dir():
        return 0

    count = 0
    for _, _, filenames in os.walk(root):
        for filename in filenames:
            if Path(filename).suffix.casefold() in IMAGE_EXTENSIONS:
                count += 1

    return count


def load_material_jsons(root):
    materials = {}

    for path in root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue

        if data.get("_type") != "material":
            continue

        material_name = path.stem
        relative_without_extension = path.relative_to(root).with_suffix("").as_posix()
        materials[normalize_name(material_name)] = data
        materials[normalize_name(relative_without_extension)] = data
        if relative_without_extension.startswith("materials/"):
            materials[normalize_name(relative_without_extension[len("materials/") :])] = data

    return materials


def find_material_data(materials, blender_material_name):
    base_name = strip_blender_suffix(blender_material_name)
    normalized_base_name = normalize_name(base_name)
    normalized_material_name = normalize_name(blender_material_name)

    return (
        materials.get(normalized_base_name)
        or materials.get(normalized_material_name)
        or materials.get(normalized_base_name.replace("\\", "/"))
        or materials.get(normalized_base_name.split("/")[-1])
    )


def find_texture(material_data, semantic_names):
    for texture in material_data.get("textures", []):
        semantic = texture.get("semantic") or texture.get("name") or ""
        if normalize_name(semantic) in semantic_names:
            return texture.get("image")

    return None


def find_image_path(images, image_name):
    if not image_name:
        return None

    for key in image_lookup_keys(image_name):
        image_path = images.get(key)
        if image_path:
            return image_path

    return None


def load_image(image_path):
    image_path = str(image_path)
    for image in bpy.data.images:
        if bpy.path.abspath(image.filepath) == image_path:
            return image

    return bpy.data.images.load(image_path, check_existing=True)


def get_or_create_node(nodes, node_type, label):
    for node in nodes:
        if node.bl_idname == node_type and node.label == label:
            return node

    node = nodes.new(node_type)
    node.label = label
    return node


def link_once(links, from_socket, to_socket):
    for link in links:
        if link.from_socket == from_socket and link.to_socket == to_socket:
            return

    links.new(from_socket, to_socket)


def replace_input_link(links, from_socket, to_socket):
    for link in list(links):
        if link.to_socket == to_socket:
            links.remove(link)

    links.new(from_socket, to_socket)


def configure_material(material, material_data, images):
    material.use_nodes = True
    material.blend_method = "OPAQUE"
    material.show_transparent_back = False
    material.diffuse_color = (material.diffuse_color[0], material.diffuse_color[1], material.diffuse_color[2], 1.0)

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        for node in nodes:
            if node.bl_idname == "ShaderNodeBsdfPrincipled":
                bsdf = node
                break
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    alpha_input = bsdf.inputs.get("Alpha")
    if alpha_input is not None:
        alpha_input.default_value = 1.0

    applied = 0
    missing = []

    color_image_name = find_texture(material_data, {normalize_name("colorMap"), normalize_name("diffuseMap"), normalize_name("albedoMap")}) if material_data else None
    color_image_path = find_image_path(images, color_image_name)
    if not color_image_path:
        color_image_name = strip_blender_suffix(material.name)
        color_image_path = find_image_path(images, color_image_name)
    if color_image_path:
        color_node = get_or_create_node(nodes, "ShaderNodeTexImage", "OpenAssetTools colorMap")
        color_node.name = "OpenAssetTools colorMap"
        color_node.image = load_image(color_image_path)
        color_node.image.colorspace_settings.name = "sRGB"
        base_color_input = bsdf.inputs.get("Base Color") or bsdf.inputs.get("Diffuse Color")
        if base_color_input is not None:
            replace_input_link(links, color_node.outputs["Color"], base_color_input)
        material.diffuse_color = (1.0, 1.0, 1.0, 1.0)

        applied += 1
    elif color_image_name:
        missing.append(f"{material.name}: {color_image_name}")

    normal_image_name = find_texture(material_data, {normalize_name("normalMap"), normalize_name("normal")}) if material_data else None
    normal_image_path = find_image_path(images, normal_image_name)
    if normal_image_path:
        normal_tex_node = get_or_create_node(nodes, "ShaderNodeTexImage", "OpenAssetTools normalMap")
        normal_tex_node.name = "OpenAssetTools normalMap"
        normal_tex_node.image = load_image(normal_image_path)
        normal_tex_node.image.colorspace_settings.name = "Non-Color"

        normal_map_node = get_or_create_node(nodes, "ShaderNodeNormalMap", "OpenAssetTools normal")
        normal_map_node.name = "OpenAssetTools normal"
        replace_input_link(links, normal_tex_node.outputs["Color"], normal_map_node.inputs["Color"])
        replace_input_link(links, normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])
        applied += 1
    elif normal_image_name:
        missing.append(f"{material.name}: {normal_image_name}")

    return applied, missing


class IMPORT_OT_openassettools_materials(Operator, ImportHelper):
    bl_idname = "import_scene.openassettools_materials"
    bl_label = "Import OpenAssetTools Materials"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""

    directory: StringProperty(
        name="Asset Root",
        description="Folder to recursively search for material JSON files and texture images",
        subtype="DIR_PATH",
    )

    selected_only: BoolProperty(
        name="Selected Objects Only",
        description="Only populate materials used by selected mesh objects",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "directory")
        layout.prop(self, "selected_only")

    def execute(self, context):
        root = Path(bpy.path.abspath(self.directory))
        if not root.is_dir():
            self.report({"ERROR"}, "Select a valid asset root folder")
            return {"CANCELLED"}

        materials = load_material_jsons(root)
        asset_root_image_count = count_image_files(root)
        images = index_images(root)

        objects = context.selected_objects if self.selected_only else context.scene.objects
        blender_materials = {
            slot.material.name: slot.material
            for obj in objects
            if obj.type == "MESH"
            for slot in obj.material_slots
            if slot.material
        }

        matched = 0
        name_fallback_matched = 0
        populated = 0
        missing_images = []
        for material in blender_materials.values():
            material_data = find_material_data(materials, material.name)
            if material_data:
                matched += 1
            else:
                name_fallback_matched += 1

            applied, missing = configure_material(material, material_data, images)
            populated += applied
            missing_images.extend(missing)

        if missing_images:
            missing_text = ", ".join(sorted(set(missing_images))[:8])
            self.report(
                {"WARNING"},
                f"Scanned {asset_root_image_count} images recursively. "
                f"Matched {matched} material JSONs and tried {name_fallback_matched} name fallbacks. "
                f"Assigned {populated} texture nodes, missing {len(missing_images)} images: {missing_text}",
            )
        else:
            self.report(
                {"INFO"},
                f"Scanned {asset_root_image_count} images recursively. "
                f"Matched {matched} material JSONs, tried {name_fallback_matched} name fallbacks, and assigned {populated} texture nodes",
            )
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_openassettools_materials.bl_idname, text="OpenAssetTools Materials")


def register():
    bpy.utils.register_class(IMPORT_OT_openassettools_materials)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_OT_openassettools_materials)


if __name__ == "__main__":
    register()
