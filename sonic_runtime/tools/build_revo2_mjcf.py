#!/usr/bin/env python3
"""Build G1+REVO2 MJCF variants without modifying the DEX3 source models.

The supplied BrainCo hand MJCF files are standalone models.  This generator
merges their defaults, meshes, collision geoms, contact exclusions, equality
constraints and six actuators into each G1 model copy.  Generated variants and
copied mesh assets live next to the original DEX3 files.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVO2_ROOT = Path("/home/url/Downloads/Revo2_xml")
MODEL_DIRS = (
    PROJECT_ROOT / "sonic_runtime/gear_sonic/data/robot_model/model_data/g1",
    PROJECT_ROOT / "sonic_runtime/gear_sonic_deploy/g1",
)

MOUNT = {
    # REVO2 fingers extend along source +Z.  G1 hands extend along wrist +X.
    # The mirrored rotations also preserve index-to-pinky ordering along G1 Z.
    "left": ("0.0415 0.003 0", "0.5 -0.5 0.5 -0.5"),
    "right": ("0.0415 -0.003 0", "0.5 0.5 0.5 0.5"),
}

COMMAND_SUFFIXES = (
    "thumb_metacarpal_joint",
    "thumb_proximal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"body not found: {name}")
    return body


def _remove_dex3(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            name = child.get("name", "")
            if name.startswith(("left_hand_", "right_hand_")):
                asset.remove(child)

    for side in ("left", "right"):
        wrist = _find_body(root, f"{side}_wrist_yaw_link")
        for child in list(wrist):
            if child.tag == "body" and child.get("name", "").startswith(f"{side}_hand_"):
                wrist.remove(child)
            elif child.tag == "geom" and child.get("mesh", "").startswith(f"{side}_hand_"):
                wrist.remove(child)

    actuator = root.find("actuator")
    if actuator is not None:
        for child in list(actuator):
            if child.get("joint", "").startswith(("left_hand_", "right_hand_")):
                actuator.remove(child)

    sensor = root.find("sensor")
    if sensor is not None:
        for child in list(sensor):
            refs = " ".join(child.attrib.values())
            if "left_hand_" in refs or "right_hand_" in refs:
                sensor.remove(child)


def _rename_references(element: ET.Element, mapping: dict[str, str]) -> None:
    for node in element.iter():
        for attribute in ("mesh", "material", "class", "childclass"):
            value = node.get(attribute)
            if value in mapping:
                node.set(attribute, mapping[value])


def _prefix_named_visuals(element: ET.Element, side: str) -> None:
    for node in element.iter():
        if node.tag in ("geom", "site") and node.get("name"):
            node.set("name", f"revo2_{side}_{node.get('name')}")


def _insert_section_before(root: ET.Element, tag: str, before: str) -> ET.Element:
    existing = root.find(tag)
    if existing is not None:
        return existing
    section = ET.Element(tag)
    children = list(root)
    index = next((i for i, child in enumerate(children) if child.tag == before), len(children))
    root.insert(index, section)
    return section


def _copy_hand_assets(
    g1_root: ET.Element,
    hand_root: ET.Element,
    hand_xml: Path,
    model_dir: Path,
    side: str,
) -> dict[str, str]:
    target_asset = g1_root.find("asset")
    source_asset = hand_root.find("asset")
    if target_asset is None or source_asset is None:
        raise RuntimeError("asset section missing")

    mapping: dict[str, str] = {}
    asset_dir = model_dir / "meshes" / f"revo2_{side}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    for source in source_asset:
        copied = copy.deepcopy(source)
        old_name = copied.get("name")
        if old_name:
            new_name = f"revo2_{side}_{old_name}"
            mapping[old_name] = new_name
            copied.set("name", new_name)
        mesh_file = source.get("file")
        if mesh_file:
            source_file = (hand_xml.parent / mesh_file).resolve()
            if not source_file.is_file():
                raise FileNotFoundError(source_file)
            destination = asset_dir / source_file.name
            shutil.copy2(source_file, destination)
            copied.set("file", f"revo2_{side}/{source_file.name}")
        target_asset.append(copied)
    return mapping


def _merge_hand(
    g1_root: ET.Element,
    hand_xml: Path,
    model_dir: Path,
    side: str,
) -> None:
    hand_root = ET.parse(hand_xml).getroot()
    mapping = _copy_hand_assets(g1_root, hand_root, hand_xml, model_dir, side)

    source_default = hand_root.find("default/default")
    target_default = g1_root.find("default")
    if source_default is None or target_default is None:
        raise RuntimeError(f"{side} hand default section missing")
    default_copy = copy.deepcopy(source_default)
    for node in default_copy.iter():
        class_name = node.get("class")
        if class_name:
            renamed = f"revo2_{side}_{class_name}"
            mapping[class_name] = renamed
            node.set("class", renamed)
    _rename_references(default_copy, mapping)
    target_default.append(default_copy)

    source_body = hand_root.find("worldbody/body")
    if source_body is None:
        raise RuntimeError(f"{side} hand root body missing")
    body_copy = copy.deepcopy(source_body)
    body_copy.set("pos", MOUNT[side][0])
    body_copy.set("quat", MOUNT[side][1])

    # The supplied right model contains five additional free tip hinges while
    # the left model uses fixed tips.  Remove those hinges so both hands have
    # the same 11 physical joints and the tips remain collision geometry.
    for parent in body_copy.iter("body"):
        for joint in list(parent.findall("joint")):
            if "tip" in joint.get("name", ""):
                parent.remove(joint)

    _rename_references(body_copy, mapping)
    _prefix_named_visuals(body_copy, side)
    _find_body(g1_root, f"{side}_wrist_yaw_link").append(body_copy)

    source_contact = hand_root.find("contact")
    if source_contact is not None:
        target_contact = _insert_section_before(g1_root, "contact", "worldbody")
        for exclusion in source_contact:
            target_contact.append(copy.deepcopy(exclusion))

    target_actuator = g1_root.find("actuator")
    if target_actuator is None:
        raise RuntimeError("G1 actuator section missing")
    force_limits = (0.5, 1.1, 2.0, 2.0, 2.0, 2.0)
    for suffix, force_limit in zip(COMMAND_SUFFIXES, force_limits):
        joint_name = f"{side}_{suffix}"
        target_actuator.append(
            ET.Element(
                "motor",
                {
                    "name": f"{side}_revo2_{suffix.removesuffix('_joint')}",
                    "joint": joint_name,
                    "ctrlrange": f"-{force_limit:g} {force_limit:g}",
                },
            )
        )

    source_equality = hand_root.find("equality")
    if source_equality is not None:
        target_equality = _insert_section_before(g1_root, "equality", "sensor")
        for constraint in source_equality:
            target_equality.append(copy.deepcopy(constraint))

    target_sensor = g1_root.find("sensor")
    if target_sensor is not None:
        joint_names = [
            f"{side}_{suffix}"
            for suffix in COMMAND_SUFFIXES
            + (
                "thumb_distal_joint",
                "index_distal_joint",
                "middle_distal_joint",
                "ring_distal_joint",
                "pinky_distal_joint",
            )
        ]
        for joint_name in joint_names:
            target_sensor.append(
                ET.Element("jointpos", {"name": f"{joint_name}_pos", "joint": joint_name})
            )
            target_sensor.append(
                ET.Element("jointvel", {"name": f"{joint_name}_vel", "joint": joint_name})
            )


def _write_model(model_dir: Path, revo2_root: Path) -> Path:
    source = model_dir / "g1_29dof_with_hand.xml"
    destination = model_dir / "g1_29dof_with_revo2.xml"
    root = ET.parse(source).getroot()
    root.set("model", "g1_29dof_with_revo2")
    root.insert(
        0,
        ET.Comment(
            "Generated by sonic_runtime/tools/build_revo2_mjcf.py; "
            "g1_29dof_with_hand.xml remains the DEX3 source/backup."
        ),
    )
    _remove_dex3(root)
    _merge_hand(
        root,
        revo2_root / "xml_left/brainco-lefthand-v2.xml",
        model_dir,
        "left",
    )
    _merge_hand(
        root,
        revo2_root / "xml_right/brainco-righthand-v2.xml",
        model_dir,
        "right",
    )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=False)
    return destination


def _scene_destination(source: Path) -> Path:
    stem = source.stem
    if stem.endswith("_with_hand"):
        stem = stem.removesuffix("_with_hand") + "_with_revo2"
    else:
        stem += "_revo2"
    return source.with_name(stem + source.suffix)


def _write_scenes(model_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for source in sorted(model_dir.glob("scene*.xml")):
        if "revo2" in source.name:
            continue
        text = source.read_text()
        if 'include file="g1_29dof_with_hand.xml"' not in text:
            continue
        destination = _scene_destination(source)
        text = text.replace(
            'include file="g1_29dof_with_hand.xml"',
            'include file="g1_29dof_with_revo2.xml"',
            1,
        )
        text = text.replace("DEX3", "BrainCo REVO2")
        destination.write_text(text)
        outputs.append(destination)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revo2-root", type=Path, default=DEFAULT_REVO2_ROOT)
    args = parser.parse_args()
    for model_dir in MODEL_DIRS:
        model = _write_model(model_dir, args.revo2_root.resolve())
        print(model)
        for scene in _write_scenes(model_dir):
            print(scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
