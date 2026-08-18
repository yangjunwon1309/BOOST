"""RGB-D and SAM3 overlay rendering for the live web view."""

from __future__ import annotations

import cv2
import numpy as np

PALETTE_BGR = np.asarray(
    [
        [64, 64, 255],
        [96, 220, 64],
        [255, 128, 64],
        [48, 190, 255],
        [255, 80, 190],
        [220, 220, 48],
    ],
    dtype=np.float32,
)


def render_segmented_rgbd(
    image_bgr: np.ndarray,
    depth_raw: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    depth_scale_um: int,
    min_depth_m: float = 0.15,
    max_depth_m: float = 3.0,
    *,
    include_depth_panel: bool = True,
    draw_score: bool = True,
    draw_depth_text: bool = True,
    return_instance_colors: bool = False,
) -> tuple[np.ndarray, list[dict]] | tuple[np.ndarray, list[dict], list[tuple[int, int, int]]]:
    """Render segmentation with optional reviewer-facing presentation details.

    The local diagnostic view retains the score, depth inset, and depth text by
    default.  The remote reviewer view can remove those diagnostics and obtain
    the exact mask tint used for each instance.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be HxWx3")
    height, width = image_bgr.shape[:2]
    if depth_raw.shape != (height, width):
        raise ValueError("depth must already be at RGB resolution")
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError("masks must be NxHxW")
    if len(masks) != len(boxes) or len(masks) != len(scores) or len(masks) != len(labels):
        raise ValueError("mask, box, score, and label counts differ")

    depth_m = depth_raw.astype(np.float32) * (depth_scale_um / 1_000_000.0)
    valid_depth = (
        (depth_raw != 0) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    )
    canvas = image_bgr.astype(np.float32).copy()
    objects: list[dict] = []
    instance_colors: list[tuple[int, int, int]] = []
    for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask = mask.astype(bool)
        color = PALETTE_BGR[index % len(PALETTE_BGR)]
        instance_colors.append(tuple(int(value) for value in color))
        canvas[mask] = canvas[mask] * 0.52 + color * 0.48
        object_depth = depth_m[mask & valid_depth]
        median_depth = (
            float(np.median(object_depth)) if object_depth.size else None
        )
        objects.append(
            {
                "index": index,
                "label": labels[index],
                "score": float(score),
                "median_depth_m": median_depth,
                "mask_pixels": int(mask.sum()),
                "box_xyxy": [float(value) for value in box],
            }
        )

    output = np.clip(canvas, 0, 255).astype(np.uint8)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        color = tuple(int(value) for value in PALETTE_BGR[index % len(PALETTE_BGR)])
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
        x0, x1 = np.clip([x0, x1], 0, width - 1)
        y0, y1 = np.clip([y0, y1], 0, height - 1)
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        if draw_score or draw_depth_text:
            median = objects[index]["median_depth_m"]
            score_text = f" {float(score):.2f}" if draw_score else ""
            depth_text = (
                "" if not draw_depth_text or median is None else f" {median:.2f}m"
            )
            cv2.putText(
                output,
                f"{labels[index]}{score_text}{depth_text}",
                (x0, max(18, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    if include_depth_panel:
        normalized = np.zeros((height, width), dtype=np.uint8)
        if np.any(valid_depth):
            normalized[valid_depth] = np.clip(
                (depth_m[valid_depth] - min_depth_m)
                * 255.0
                / max(1e-6, max_depth_m - min_depth_m),
                0,
                255,
            ).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        depth_color[~valid_depth] = 0
        inset_width = max(160, width // 3)
        inset_height = max(90, round(inset_width * height / width))
        inset_width = min(inset_width, width - 16)
        inset_height = min(inset_height, height - 16)
        depth_color = cv2.resize(
            depth_color, (inset_width, inset_height), interpolation=cv2.INTER_NEAREST
        )
        left, top = width - inset_width - 8, height - inset_height - 8
        output[top : top + inset_height, left : left + inset_width] = depth_color
        cv2.rectangle(
            output, (left, top), (left + inset_width, top + inset_height), (255, 255, 255), 1)
        cv2.putText(
            output,
            "aligned depth",
            (left + 6, top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if return_instance_colors:
        return output, objects, instance_colors
    return output, objects
