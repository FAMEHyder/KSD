
import os
import io
import json
import base64
import traceback

import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from torchvision import models, transforms

from scipy.spatial.distance import pdist
from segment_anything import sam_model_registry, SamPredictor


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")

YOLO_MODEL_PATH = os.path.join(
    MODELS_DIR, "epoch71.pt"
)

SAM_CHECKPOINT_PATH = os.path.join(
    MODELS_DIR, "sam_vit_b_01ec64.pth"
)

GATEKEEPER_MODEL_PATH = os.path.join(
    MODELS_DIR, "MobileNet_Gatekeeper.pth"
)

CLASS_NAMES_PATH = os.path.join(
    MODELS_DIR, "class_names.json"
)

SAM_MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.25

GATEKEEPER_THRESHOLD = 0.60

GATEKEEPER_IMAGE_SIZE = 224

MAX_IMAGE_SIZE = 10 * 1024 * 1024

PIXEL_SPACING_MM = 0.75

GATEKEEPER_CLASSES = [
    "brain_ct",
    "chest_ct",
    "kidney_ct",
    "mri",
    "nonmedical",
    "normal_human",
    "other_body_ct",
    "ultrasound",
    "xray"
]


# =========================================================
# 2. MOBILENET PREPROCESSING
# =========================================================

GATEKEEPER_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (GATEKEEPER_IMAGE_SIZE, GATEKEEPER_IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 3. FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    },
)


# =========================================================
# 4. LOAD MOBILENET GATEKEEPER
# =========================================================

print("Loading MobileNetV3 Gatekeeper...")

if not os.path.exists(GATEKEEPER_MODEL_PATH):
    raise FileNotFoundError(
        f"Gatekeeper model not found: {GATEKEEPER_MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8-sig"
) as file:

    loaded_classes = json.load(file)


if loaded_classes != GATEKEEPER_CLASSES:

    raise ValueError(
        "class_names.json does not match the expected "
        "MobileNet class order. Check train_ds.classes "
        "and the saved model's training configuration."
    )


gatekeeper_model = models.mobilenet_v3_large(
    weights=None
)

in_features = gatekeeper_model.classifier[3].in_features

gatekeeper_model.classifier[3] = torch.nn.Linear(
    in_features,
    len(GATEKEEPER_CLASSES)
)


try:

    checkpoint = torch.load(
        GATEKEEPER_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

except TypeError:

    checkpoint = torch.load(
        GATEKEEPER_MODEL_PATH,
        map_location=DEVICE
    )


if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]


gatekeeper_model.load_state_dict(
    checkpoint,
    strict=True
)

gatekeeper_model.to(DEVICE)

gatekeeper_model.eval()

print("MobileNetV3 Gatekeeper loaded successfully.")


# =========================================================
# 5. LOAD YOLO MODEL
# =========================================================

print("Loading YOLOv11 model...")

if not os.path.exists(YOLO_MODEL_PATH):

    raise FileNotFoundError(
        f"YOLO model not found: {YOLO_MODEL_PATH}"
    )


yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLOv11 model loaded successfully.")


# =========================================================
# 6. LOAD ORIGINAL SAM MODEL
# =========================================================

print(f"Loading Original SAM ViT-B on {DEVICE}...")

if not os.path.exists(SAM_CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"SAM checkpoint not found: {SAM_CHECKPOINT_PATH}"
    )


sam_model = sam_model_registry[SAM_MODEL_TYPE](
    checkpoint=SAM_CHECKPOINT_PATH
)

sam_model.to(device=DEVICE)

sam_model.eval()

sam_predictor = SamPredictor(sam_model)

print("Original SAM loaded successfully.")

print(f"All models ready. Device: {DEVICE}")


# =========================================================
# 7. MOBILENET GATEKEEPER PREDICTION
# =========================================================

def verify_kidney_ct(image):

    image_tensor = GATEKEEPER_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = gatekeeper_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = int(predicted_index.item())

    predicted_class = GATEKEEPER_CLASSES[predicted_index]

    predicted_probability = float(confidence.item())

    kidney_index = GATEKEEPER_CLASSES.index(
        "kidney_ct"
    )

    kidney_probability = float(
        probabilities[0][kidney_index].item()
    )

    accepted = (
        predicted_class == "kidney_ct"
        and kidney_probability >= GATEKEEPER_THRESHOLD
    )

    return {

        "accepted": accepted,

        "predicted_class": predicted_class,

        "confidence": round(
            predicted_probability * 100,
            2
        ),

        "kidney_ct_probability": round(
            kidney_probability * 100,
            2
        ),

        "threshold": GATEKEEPER_THRESHOLD * 100

    }


# =========================================================
# 8. IMAGE CONVERSION
# =========================================================

def image_to_data_uri(image_array):

    image = Image.fromarray(
        image_array.astype(np.uint8)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# 9. BOX VALIDATION
# =========================================================

def clamp_box(box, image_width, image_height):

    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(x1, image_width - 1))

    y1 = max(0, min(y1, image_height - 1))

    x2 = max(0, min(x2, image_width - 1))

    y2 = max(0, min(y2, image_height - 1))

    if x2 <= x1 or y2 <= y1:

        return None

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32
    )


# =========================================================
# 10. ORIGINAL SAM SEGMENTATION
# =========================================================

def sam_segment(image_array, box):

    input_box = np.array(
        box,
        dtype=np.float32
    )

    masks, scores, logits = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False
    )

    mask = masks[0].astype(np.uint8)

    sam_score = float(scores[0])

    return mask, sam_score


# =========================================================
# 11. DETAILED STONE MEASUREMENT
# =========================================================

def measure_stone(mask, box, pixel_spacing_mm=PIXEL_SPACING_MM):
    """
    Computes exact clinical stone metrics:
    - Maximum caliper distance (Feret's diameter, primary clinical metric for stone size)
    - Transverse width (minor axis)
    - Surface area (px and mm²)
    - Prolate ellipsoid volume estimate (mm³)
    - Caliper endpoints for drawing PACS measurement lines
    - Clinical category and risk stratification
    """
    x1, y1, x2, y2 = box
    box_w_px = max(1.0, float(x2 - x1))
    box_h_px = max(1.0, float(y2 - y1))

    y_coords, x_coords = np.where(mask > 0)
    has_mask = len(x_coords) > 0

    caliper_p1 = None
    caliper_p2 = None

    if has_mask:
        x_min = int(x_coords.min())
        x_max = int(x_coords.max())
        y_min = int(y_coords.min())
        y_max = int(y_coords.max())

        mask_w_px = float(x_max - x_min + 1)
        mask_h_px = float(y_max - y_min + 1)
        area_px = int(np.count_nonzero(mask))

        points = np.column_stack((x_coords, y_coords)).astype(np.float32)
        hull = cv2.convexHull(points)
        hull_points = hull.reshape(-1, 2)

        if len(hull_points) > 1:
            diff = hull_points[:, np.newaxis, :] - hull_points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            max_i, max_j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            max_distance_px = float(dist_matrix[max_i, max_j])
            caliper_p1 = (int(round(hull_points[max_i][0])), int(round(hull_points[max_i][1])))
            caliper_p2 = (int(round(hull_points[max_j][0])), int(round(hull_points[max_j][1])))
        else:
            max_distance_px = float(max(mask_w_px, mask_h_px))
            caliper_p1 = (int(x_min), int(y_min))
            caliper_p2 = (int(x_max), int(y_max))

        if len(points) >= 5:
            rect = cv2.minAreaRect(points)
            (rx, ry), (rw, rh), angle = rect
            major_axis_px = float(max(rw, rh))
            minor_axis_px = float(min(rw, rh))
        else:
            major_axis_px = max_distance_px
            minor_axis_px = float(min(mask_w_px, mask_h_px))

        length_px = max(max_distance_px, major_axis_px)
        width_px = max(1.0, minor_axis_px)
    else:
        # Fallback to YOLO bounding box if SAM returned no pixels
        length_px = max(box_w_px, box_h_px)
        width_px = min(box_w_px, box_h_px)
        area_px = int(box_w_px * box_h_px * 0.75)
        max_distance_px = length_px
        caliper_p1 = (int(x1), int(y1))
        caliper_p2 = (int(x2), int(y2))

    length_mm = length_px * pixel_spacing_mm
    width_mm = width_px * pixel_spacing_mm
    area_mm2 = area_px * (pixel_spacing_mm ** 2)
    # Prolate ellipsoid: (pi / 6) * length * width^2
    volume_mm3 = (np.pi / 6.0) * length_mm * (width_mm ** 2)

    # Clinical categorization based on EAU / AUA Urology Guidelines
    if length_mm < 5.0:
        clinical_category = "Small (< 5 mm)"
        risk_level = "Low Risk"
        spontaneous_passage = "High (~80-90%)"
        clinical_action = "Conservative observation, hydration, and analgesics typically sufficient."
    elif length_mm <= 10.0:
        clinical_category = "Moderate (5 - 10 mm)"
        risk_level = "Moderate Risk"
        spontaneous_passage = "Moderate (~50%)"
        clinical_action = "Medical Expulsive Therapy (MET) or Shock Wave Lithotripsy (ESWL) may be recommended."
    else:
        clinical_category = "Large (> 10 mm)"
        risk_level = "High Risk"
        spontaneous_passage = "Low (< 20%)"
        clinical_action = "Urological intervention (e.g. Ureteroscopy / PCNL) typically required to prevent obstruction."

    return {
        "area_px": int(area_px),
        "width_px": round(width_px, 1),
        "height_px": round(length_px, 1),
        "max_distance_px": round(max_distance_px, 2),
        "estimated_size_mm": round(length_mm, 2),
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "area_mm2": round(area_mm2, 2),
        "volume_mm3": round(volume_mm3, 2),
        "caliper_p1": list(caliper_p1) if caliper_p1 else None,
        "caliper_p2": list(caliper_p2) if caliper_p2 else None,
        "clinical_category": clinical_category,
        "risk_level": risk_level,
        "spontaneous_passage": spontaneous_passage,
        "clinical_action": clinical_action,
    }


def measure_mask(mask):
    """
    Measures the combined mask for total stone burden.
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return {
            "area_px": 0,
            "width_px": 0,
            "height_px": 0,
            "max_distance_px": 0,
            "estimated_size_mm": 0,
            "length_mm": 0,
            "width_mm": 0,
            "area_mm2": 0,
            "volume_mm3": 0,
        }
    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
    hull = cv2.convexHull(points)
    hull_points = hull.reshape(-1, 2)
    if len(hull_points) > 1:
        max_dist = float(pdist(hull_points).max())
    else:
        max_dist = 0.0
    area_px = int(np.count_nonzero(mask))
    w_px = float(x_coords.max() - x_coords.min() + 1)
    h_px = float(y_coords.max() - y_coords.min() + 1)
    length_mm = max_dist * PIXEL_SPACING_MM
    width_mm = min(w_px, h_px) * PIXEL_SPACING_MM
    area_mm2 = area_px * (PIXEL_SPACING_MM ** 2)
    volume_mm3 = (np.pi / 6.0) * length_mm * (width_mm ** 2)
    return {
        "area_px": area_px,
        "width_px": round(w_px, 1),
        "height_px": round(h_px, 1),
        "max_distance_px": round(max_dist, 2),
        "estimated_size_mm": round(length_mm, 2),
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "area_mm2": round(area_mm2, 2),
        "volume_mm3": round(volume_mm3, 2),
    }


# =========================================================
# 12. DRAW YOLO OUTPUT IMAGES
# =========================================================

def draw_yolo_images(image_array, detections):
    """
    Draws clear bounding boxes and exact size labels on YOLO images.
    """
    plain_image = Image.fromarray(image_array.copy())
    labeled_image = Image.fromarray(image_array.copy())

    plain_draw = ImageDraw.Draw(plain_image)
    labeled_draw = ImageDraw.Draw(labeled_image)

    try:
        font_large = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        box = (int(x1), int(y1), int(x2), int(y2))

        # Color: Vibrant Cyan for high contrast on CT grayscale
        box_color = (0, 229, 255)

        # 1. Draw Bounding Box (Boxed stone!)
        plain_draw.rectangle(box, outline=box_color, width=3)
        labeled_draw.rectangle(box, outline=box_color, width=3)

        # Corner accent marks on labeled image
        corner_len = min(12, max(4, int((x2 - x1) / 3)), max(4, int((y2 - y1) / 3)))
        for d in [plain_draw, labeled_draw]:
            # Top-left corner
            d.line([(x1, y1), (x1 + corner_len, y1)], fill=(255, 255, 0), width=4)
            d.line([(x1, y1), (x1, y1 + corner_len)], fill=(255, 255, 0), width=4)
            # Bottom-right corner
            d.line([(x2, y2), (x2 - corner_len, y2)], fill=(255, 255, 0), width=4)
            d.line([(x2, y2), (x2, y2 - corner_len)], fill=(255, 255, 0), width=4)

        # 2. Extract size information
        dims = detection.get("dimensions", {})
        size_mm = dims.get("estimated_size_mm", 0.0)
        conf = detection.get("confidence", 0.0)

        # Labels
        label_line1 = f"Stone #{idx + 1}: {conf:.1f}%"
        label_line2 = f"Exact Size: {size_mm:.2f} mm" if size_mm > 0 else f"{detection.get('class_name', 'Stone')}"

        # On plain_image, draw a compact badge with exact size
        badge_text = f"Stone #{idx + 1}: {size_mm:.2f} mm" if size_mm > 0 else f"Stone #{idx + 1}"
        text_y_compact = max(4, int(y1) - 22)
        tb = plain_draw.textbbox((int(x1), text_y_compact), badge_text, font=font_small)
        plain_draw.rectangle((tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2), fill=(15, 23, 42), outline=box_color, width=1)
        plain_draw.text((int(x1), text_y_compact), badge_text, fill=(0, 229, 255), font=font_small)

        # On labeled_image, draw complete detailed badge with confidence and exact size
        text_y = max(4, int(y1) - 44)
        tb1 = labeled_draw.textbbox((int(x1), text_y), label_line1, font=font_small)
        tb2 = labeled_draw.textbbox((int(x1), text_y + 18), label_line2, font=font_large)
        box_w = max(tb1[2], tb2[2]) + 8
        box_h = tb2[3] + 3
        labeled_draw.rectangle((int(x1) - 2, text_y - 2, box_w, box_h), fill=(15, 23, 42), outline=(0, 229, 255), width=1)
        labeled_draw.text((int(x1) + 2, text_y), label_line1, fill=(203, 213, 225), font=font_small)
        labeled_draw.text((int(x1) + 2, text_y + 18), label_line2, fill=(255, 215, 0), font=font_large)

    return (
        np.array(plain_image),
        np.array(labeled_image),
    )


# =========================================================
# 13. SEGMENTATION OVERLAY
# =========================================================

def make_segmentation_overlay(
    image_array,
    combined_mask,
    detections=None
):

    overlay = image_array.copy()

    mask = combined_mask.astype(bool)

    if np.any(mask):

        green_layer = np.zeros_like(image_array)

        green_layer[:, :] = (16, 185, 129)

        alpha = 0.42

        overlay[mask] = (
            image_array[mask] * (1 - alpha)
            + green_layer[mask] * alpha
        ).astype(np.uint8)

        mask_uint8 = (
            mask.astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (239, 68, 68),
            2,
        )

    # Box the detected stones
    if detections:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 229, 255), 2)

    return overlay


# =========================================================
# 14. SEGMENTATION IMAGE WITH BOUNDING BOX & EXACT SIZE
# =========================================================

def make_size_labeled_image(
    image_array,
    detections,
    combined_mask
):

    output = make_segmentation_overlay(
        image_array,
        combined_mask,
        detections
    )

    output_image = Image.fromarray(output)

    draw = ImageDraw.Draw(output_image)

    try:

        font_title = ImageFont.truetype(
            "arial.ttf",
            16
        )

        font_sub = ImageFont.truetype(
            "arial.ttf",
            13
        )

        font_header = ImageFont.truetype(
            "arial.ttf",
            13
        )

    except OSError:

        font_title = ImageFont.load_default()

        font_sub = ImageFont.load_default()

        font_header = ImageFont.load_default()

    for index, detection in enumerate(detections):

        x1, y1, x2, y2 = [int(v) for v in detection["box"]]

        dims = detection.get("dimensions", {})

        size_mm = dims.get("estimated_size_mm", 0.0)

        len_mm = dims.get("length_mm", size_mm)

        w_mm = dims.get("width_mm", 0.0)

        area_mm2 = dims.get("area_mm2", 0.0)

        # 1. Bounding Box around the stone (Clear Boxed Stone!)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=2)

        # 2. Draw PACS-style Caliper measurement line across the longest stone diameter
        p1 = dims.get("caliper_p1")
        p2 = dims.get("caliper_p2")
        if p1 and p2:
            cp1 = (int(p1[0]), int(p1[1]))
            cp2 = (int(p2[0]), int(p2[1]))
            # Caliper line in bright magenta
            draw.line([cp1, cp2], fill=(244, 63, 94), width=2)
            # Caliper crosshair tick marks at endpoints
            tick_size = 4
            draw.line([(cp1[0] - tick_size, cp1[1]), (cp1[0] + tick_size, cp1[1])], fill=(255, 255, 255), width=2)
            draw.line([(cp1[0], cp1[1] - tick_size), (cp1[0], cp1[1] + tick_size)], fill=(255, 255, 255), width=2)
            draw.line([(cp2[0] - tick_size, cp2[1]), (cp2[0] + tick_size, cp2[1])], fill=(255, 255, 255), width=2)
            draw.line([(cp2[0], cp2[1] - tick_size), (cp2[0], cp2[1] + tick_size)], fill=(255, 255, 255), width=2)

        # 3. Draw high-visibility Size Badge
        label_1 = f"Stone #{index + 1}: {size_mm:.2f} mm"
        label_2 = f"{len_mm:.1f}x{w_mm:.1f} mm | {area_mm2:.1f} mm2"

        badge_x = int(x1)
        badge_y = max(6, int(y1) - 44)

        tb1 = draw.textbbox((badge_x, badge_y), label_1, font=font_title)
        tb2 = draw.textbbox((badge_x, badge_y + 18), label_2, font=font_sub)

        bg_w = max(tb1[2], tb2[2]) + 10
        bg_h = tb2[3] + 4

        draw.rectangle([badge_x - 4, badge_y - 2, bg_w, bg_h], fill=(15, 23, 42), outline=(255, 215, 0), width=1)
        draw.text((badge_x + 2, badge_y), label_1, fill=(255, 215, 0), font=font_title)
        draw.text((badge_x + 2, badge_y + 18), label_2, fill=(226, 232, 240), font=font_sub)

    # 4. Calibration header watermark
    calib_text = f"CT Calibrated: {PIXEL_SPACING_MM:.2f} mm/px | Detected Stones: {len(detections)}"
    tb_c = draw.textbbox((10, 10), calib_text, font=font_header)
    draw.rectangle([tb_c[0] - 4, tb_c[1] - 3, tb_c[2] + 4, tb_c[3] + 3], fill=(15, 23, 42), outline=(56, 189, 248), width=1)
    draw.text((10, 10), calib_text, fill=(56, 189, 248), font=font_header)

    return np.array(output_image)


# =========================================================
# 15. HOME ROUTE
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "message": "Kidney Stone Detection API is running.",

        "device": DEVICE,

    })


# =========================================================
# 16. HEALTH CHECK
# =========================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "success",

        "gatekeeper_loaded": gatekeeper_model is not None,

        "yolo_loaded": yolo_model is not None,

        "sam_loaded": sam_predictor is not None,

        "gatekeeper_threshold": GATEKEEPER_THRESHOLD,

        "gatekeeper_classes": GATEKEEPER_CLASSES,

        "device": DEVICE,

    })


# =========================================================
# 17. PREDICTION API
# =========================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # ---------------------------------------------
        # CHECK UPLOADED FILE
        # ---------------------------------------------

        if "file" not in request.files:

            return jsonify({

                "error": (
                    "No image file received. "
                    "Please upload an image."
                )

            }), 400

        uploaded_file = request.files["file"]

        if uploaded_file.filename == "":

            return jsonify({

                "error": "No file selected."

            }), 400

        # ---------------------------------------------
        # READ IMAGE
        # ---------------------------------------------

        try:

            pil_image = Image.open(
                uploaded_file.stream
            )

            pil_image = ImageOps.exif_transpose(
                pil_image
            ).convert("RGB")

        except Exception:

            return jsonify({

                "error": "Invalid or unsupported image file."

            }), 400

        image_array = np.array(
            pil_image,
            dtype=np.uint8
        )

        image_height, image_width = image_array.shape[:2]

        # ---------------------------------------------
        # STEP 1: MOBILENET GATEKEEPER
        # ---------------------------------------------

        print("Running MobileNet Gatekeeper...")

        gatekeeper_result = verify_kidney_ct(
            pil_image
        )

        print(
            "Gatekeeper Result:",
            gatekeeper_result
        )

        # ---------------------------------------------
        # STEP 2: REJECT NON-KIDNEY CT
        # ---------------------------------------------

        if not gatekeeper_result["accepted"]:

            print("Image rejected by Gatekeeper.")

            return jsonify({

                "accepted": False,

                "gatekeeper": gatekeeper_result,

                "detected": False,

                "no_stone_detected": False,

                "detection_count": 0,

                "detections": [],

                "device": DEVICE,

                "message": (
                    "Image rejected by MobileNet Gatekeeper. "
                    "Only kidney_ct images with at least "
                    "60% kidney_ct probability are accepted."
                ),

                "image": {

                    "filename": uploaded_file.filename,

                    "format": "PNG/JPG",

                    "width": image_width,

                    "height": image_height,

                },

                "original_image": image_to_data_uri(
                    image_array
                ),

                "yolo_image": None,

                "yolo_labeled_image": None,

                "segmentation_image": None,

                "segmentation_size_image": None

            }), 200

        # ---------------------------------------------
        # STEP 3: ACCEPTED KIDNEY CT
        # ---------------------------------------------

        print(
            "Kidney CT accepted by Gatekeeper."
        )

        print("Running YOLOv11 detection...")

        # ---------------------------------------------
        # STEP 4: YOLO PREDICTION
        # ---------------------------------------------

        yolo_results = yolo_model.predict(
            source=image_array,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections = []

        if yolo_results and len(yolo_results) > 0:

            result = yolo_results[0]

            if result.boxes is not None:

                boxes = result.boxes

                for index in range(len(boxes)):

                    raw_box = (
                        boxes.xyxy[index]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    valid_box = clamp_box(
                        raw_box,
                        image_width,
                        image_height,
                    )

                    if valid_box is None:

                        continue

                    class_id = int(
                        boxes.cls[index].item()
                    )

                    confidence = float(
                        boxes.conf[index].item()
                    ) * 100

                    class_name = yolo_model.names.get(
                        class_id,
                        str(class_id)
                    )

                    detections.append({

                        "box": [
                            round(float(value), 2)
                            for value in valid_box
                        ],

                        "class_id": class_id,

                        "class_name": str(class_name),

                        "confidence": round(
                            confidence,
                            2
                        ),

                        "dimensions": {

                            "area_px": 0,

                            "width_px": None,

                            "height_px": None,

                            "max_distance_px": 0,

                            "estimated_size_mm": 0

                        },

                        "sam_score": None,

                    })

        print(
            f"YOLO detections: {len(detections)}"
        )

        # ---------------------------------------------
        # STEP 5: SAM SEGMENTATION & EXACT SIZE MEASUREMENT
        # ---------------------------------------------

        print(
            "Running Original SAM segmentation & exact stone measurement..."
        )

        combined_mask = np.zeros(
            (image_height, image_width),
            dtype=np.uint8,
        )

        if detections:

            sam_predictor.set_image(
                image_array,
                image_format="RGB"
            )

            for index, detection in enumerate(detections):

                print(
                    f"SAM segmentation & measurement: "
                    f"{index + 1}/{len(detections)}"
                )

                mask, sam_score = sam_segment(
                    image_array,
                    detection["box"],
                )

                if mask.shape != (
                    image_height,
                    image_width
                ):

                    raise ValueError(
                        "SAM returned a mask with unexpected dimensions."
                    )

                detection["id"] = index + 1
                detection["sam_score"] = round(
                    sam_score,
                    4
                )

                # Compute exact stone dimensions and clinical metrics
                measurement = measure_stone(
                    mask,
                    detection["box"],
                    PIXEL_SPACING_MM
                )

                detection["dimensions"] = measurement
                detection["exact_size_mm"] = measurement["estimated_size_mm"]
                detection["length_mm"] = measurement["length_mm"]
                detection["width_mm"] = measurement["width_mm"]
                detection["area_mm2"] = measurement["area_mm2"]
                detection["volume_mm3"] = measurement["volume_mm3"]
                detection["clinical_category"] = measurement["clinical_category"]
                detection["risk_level"] = measurement["risk_level"]
                detection["spontaneous_passage"] = measurement["spontaneous_passage"]
                detection["clinical_action"] = measurement["clinical_action"]

                combined_mask = np.maximum(
                    combined_mask,
                    mask
                )

        print(
            "YOLO detection + SAM segmentation + exact size measurement completed."
        )

        # ---------------------------------------------
        # STEP 6: OUTPUT IMAGES WITH BOXES & EXACT SIZES
        # ---------------------------------------------

        # YOLO annotated images (Bounding box + exact size badge)
        yolo_image, yolo_labeled_image = draw_yolo_images(
            image_array,
            detections,
        )

        # Segmentation overlay with green mask and bounding boxes
        segmentation_overlay = make_segmentation_overlay(
            image_array,
            combined_mask,
            detections,
        )

        # Exact size measurement image with bounding box, caliper line, and metrics
        size_labeled_image = make_size_labeled_image(
            image_array,
            detections,
            combined_mask
        )

        # Combined measurements for total stone burden
        total_dimensions = measure_mask(
            combined_mask
        )

        detection_count = len(detections)

        # Primary (largest) stone identified
        primary_stone = (
            max(detections, key=lambda d: d.get("exact_size_mm", 0.0))
            if detections
            else None
        )
        primary_size_mm = (
            primary_stone["exact_size_mm"]
            if primary_stone
            else 0.0
        )

        # ---------------------------------------------
        # STEP 7: JSON RESPONSE
        # ---------------------------------------------

        response_data = {

            "accepted": True,

            "gatekeeper": gatekeeper_result,

            "detected": detection_count > 0,

            "no_stone_detected": detection_count == 0,

            "detection_count": detection_count,

            # Explicit exact size fields for frontend and reports
            "stone_size": f"{primary_size_mm:.2f} mm" if detection_count > 0 else "No stone detected",

            "estimated_size": f"{primary_size_mm:.2f} mm" if detection_count > 0 else "No stone detected",

            "size_mm": primary_size_mm,

            "primary_stone": primary_stone,

            "clinical_category": primary_stone["clinical_category"] if primary_stone else "Normal",

            "risk_level": primary_stone["risk_level"] if primary_stone else "None",

            "detections": detections,

            "device": DEVICE,

            "dimensions": total_dimensions,

            "pixel_spacing_mm": PIXEL_SPACING_MM,

            "measurement_unit": "mm",

            "measurement_note": (
                "Exact stone size is measured as the maximum caliper (Feret) diameter "
                "across the segmented stone contour calibrated at 0.75 mm/pixel. "
                "For clinical treatment planning, verify with slice-specific DICOM metadata."
            ),

            "image": {

                "filename": uploaded_file.filename,

                "format": "PNG/JPG",

                "width": image_width,

                "height": image_height,

            },

            "message": (
                f"Processing complete: {detection_count} kidney stone(s) detected with exact caliper sizing."
                if detection_count > 0
                else "Kidney CT confirmed: No kidney stone detected."
            ),

            # IMAGE RESULTS
            "original_image": image_to_data_uri(
                image_array
            ),

            "yolo_image": image_to_data_uri(
                yolo_labeled_image
            ),

            "yolo_plain_image": image_to_data_uri(
                yolo_image
            ),

            "yolo_labeled_image": image_to_data_uri(
                yolo_labeled_image
            ),

            "segmentation_image": image_to_data_uri(
                segmentation_overlay
            ),

            "segmentation_size_image": image_to_data_uri(
                size_labeled_image
            ),

        }

        return jsonify(response_data), 200

    except Exception as error:

        print(
            "Prediction error:",
            str(error)
        )

        traceback.print_exc()

        return jsonify({

            "error": str(error),

            "message": "Image processing failed.",

        }), 500


# =========================================================
# 18. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
