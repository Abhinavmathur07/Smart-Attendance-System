"""
face_recognition_module.py
Local OpenCV Haar Cascade face detection + LBPH (Local Binary Patterns Histograms)
face recognition module. Runs 100% locally without any cloud or paid API.

Known Limitations:
1. LBPH is sensitive to dramatic lighting changes, heavy shadows, and extreme head angles.
2. Requires clear frontal face visibility (glasses glare or masks reduce accuracy).
3. Does NOT include biometric liveness / anti-spoofing detection; an operator must
   visually confirm the live subject before confirming attendance.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
    CV2_FACE_AVAILABLE = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
except ImportError:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False
    CV2_FACE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
FACE_SIZE = (200, 200)


def check_opencv_environment() -> Tuple[bool, str]:
    """Verify OpenCV and opencv-contrib-python (cv2.face) availability."""
    if not CV2_AVAILABLE:
        return False, "OpenCV is not installed. Run: pip install opencv-contrib-python"
    if not CV2_FACE_AVAILABLE:
        return (
            False,
            "cv2.face module is missing. Uninstall 'opencv-python' and install 'opencv-contrib-python'.",
        )
    return True, "OpenCV LBPH recognizer ready."


def _get_face_cascade():
    ok, msg = check_opencv_environment()
    if not ok:
        raise RuntimeError(msg)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar Cascade classifier at {cascade_path}")
    return cascade


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR numpy array."""
    rgb = image.convert("RGB")
    arr = np.array(rgb)
    return arr[:, :, ::-1].copy()


def detect_faces(bgr_image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detect frontal faces in a BGR image using Haar Cascades.
    Returns (list of (x, y, w, h), grayscale image).
    """
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(70, 70),
    )
    bbox_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces] if len(faces) > 0 else []
    return bbox_list, gray


def preprocess_face(gray_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop face region, resize to uniform 200x200, and normalize histogram."""
    x, y, w, h = bbox
    roi = gray_image[y : y + h, x : x + w]
    resized = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(resized)
    return normalized


def draw_face_annotations(
    bgr_image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    label_text: str = "",
    recognized: bool = True,
) -> np.ndarray:
    """Draw bounding box and label text on a copy of the BGR image."""
    annotated = bgr_image.copy()
    color = (14, 165, 233) if recognized else (220, 38, 38)  # Cyan-blue or crimson in BGR? BGR:
    bgr_color = (233, 165, 14) if recognized else (38, 38, 220)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_color, 3)
        if label_text:
            cv2.rectangle(annotated, (x, max(0, y - 32)), (x + w, y), bgr_color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x + 8, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


def register_student_face(
    student_id: str,
    pil_image: Image.Image,
    faces_dir: Path = FACES_DIR,
) -> Tuple[bool, str, Optional[str]]:
    """
    Detect a single face in the uploaded/captured portrait, save the full portrait
    and augmented grayscale face crops for LBPH training, then retrain the model.
    """
    ok, msg = check_opencv_environment()
    if not ok:
        return False, msg, None

    sid = student_id.strip().upper()
    student_folder = faces_dir / sid
    student_folder.mkdir(parents=True, exist_ok=True)

    bgr = pil_to_bgr(pil_image)
    bboxes, gray = detect_faces(bgr)

    if len(bboxes) == 0:
        return False, "No face detected. Ensure good front lighting and look directly at the camera.", None
    if len(bboxes) > 1:
        return False, f"Multiple faces ({len(bboxes)}) detected. Please capture only one student at a time.", None

    bbox = bboxes[0]
    face_crop = preprocess_face(gray, bbox)

    # Save primary color portrait for profile display
    portrait_path = student_folder / "profile.jpg"
    pil_image.convert("RGB").save(portrait_path, format="JPEG", quality=92)

    # Save multiple subtle brightness/flip variants so LBPH trains robustly from 1 photo
    variants = [
        face_crop,
        cv2.flip(face_crop, 1),
        cv2.convertScaleAbs(face_crop, alpha=1.08, beta=8),
        cv2.convertScaleAbs(face_crop, alpha=0.92, beta=-8),
    ]
    existing = list(student_folder.glob("sample_*.jpg"))
    start_idx = len(existing) + 1
    for offset, variant in enumerate(variants):
        sample_file = student_folder / f"sample_{start_idx + offset}.jpg"
        cv2.imwrite(str(sample_file), variant)

    trained_ok, train_msg = train_lbph_model(faces_dir=faces_dir)
    if not trained_ok:
        return False, f"Face saved, but model training failed: {train_msg}", str(portrait_path)

    return True, "Face captured, preprocessed, and LBPH model updated.", str(portrait_path)


def delete_student_faces(student_id: str, faces_dir: Path = FACES_DIR) -> Tuple[bool, str]:
    """Delete all stored face images for a student and retrain the model."""
    sid = student_id.strip().upper()
    student_folder = faces_dir / sid
    if student_folder.exists():
        try:
            shutil.rmtree(student_folder)
        except OSError as exc:
            return False, f"Failed to delete face folder: {exc}"
    train_lbph_model(faces_dir=faces_dir)
    return True, f"Biometric images for {sid} permanently deleted."


def train_lbph_model(
    faces_dir: Path = FACES_DIR,
    model_path: Path = MODEL_PATH,
    label_map_path: Path = LABEL_MAP_PATH,
) -> Tuple[bool, str]:
    """Train OpenCV LBPHFaceRecognizer across all registered student face folders."""
    ok, msg = check_opencv_environment()
    if not ok:
        return False, msg

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    training_images: List[np.ndarray] = []
    training_labels: List[int] = []
    label_map: Dict[str, str] = {}

    student_dirs = sorted([d for d in faces_dir.iterdir() if d.is_dir()])
    for label_idx, s_dir in enumerate(student_dirs):
        samples = sorted(s_dir.glob("sample_*.jpg"))
        if not samples:
            continue
        label_map[str(label_idx)] = s_dir.name
        for sample_path in samples:
            img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, FACE_SIZE, interpolation=cv2.INTER_AREA)
            training_images.append(resized)
            training_labels.append(label_idx)

    if not training_images:
        if model_path.exists():
            model_path.unlink()
        if label_map_path.exists():
            label_map_path.unlink()
        return False, "No training face samples found. Register at least one student with a face photo."

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(training_images, np.array(training_labels, dtype=np.int32))
        recognizer.write(str(model_path))
        label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
        return True, f"LBPH model trained on {len(label_map)} student(s) ({len(training_images)} samples)."
    except Exception as exc:
        return False, f"LBPH training error: {exc}"


def recognize_face(
    pil_image: Image.Image,
    threshold: float = 65.0,
    model_path: Path = MODEL_PATH,
    label_map_path: Path = LABEL_MAP_PATH,
) -> Dict[str, Any]:
    """
    Detect a face in the provided image and match against the trained LBPH model.
    Lower LBPH distance indicates a closer match (0 = exact match).
    If distance <= threshold, returns recognized=True with student_id & confidence score.
    """
    ok, msg = check_opencv_environment()
    if not ok:
        return {"status": "error", "message": msg, "recognized": False}

    if not model_path.exists() or not label_map_path.exists():
        return {
            "status": "untrained",
            "message": "Face recognition model is not trained yet. Register a student with a face image first.",
            "recognized": False,
        }

    bgr = pil_to_bgr(pil_image)
    bboxes, gray = detect_faces(bgr)

    if len(bboxes) == 0:
        return {
            "status": "no_face",
            "message": "No face detected in frame. Adjust lighting and face the camera directly.",
            "recognized": False,
            "annotated_rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        }

    if len(bboxes) > 1:
        annotated = draw_face_annotations(bgr, bboxes, "Multiple Faces", recognized=False)
        return {
            "status": "multiple_faces",
            "message": f"Detected {len(bboxes)} faces. Only one student should appear in frame at a time.",
            "recognized": False,
            "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        }

    bbox = bboxes[0]
    face_crop = preprocess_face(gray, bbox)

    try:
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(model_path))
        predicted_label, distance = recognizer.predict(face_crop)
    except Exception as exc:
        return {"status": "error", "message": f"Prediction failed: {exc}", "recognized": False}

    student_id = label_map.get(str(predicted_label))
    # Convert LBPH distance (0..120+) into a readable similarity percentage
    similarity_pct = round(max(0.0, min(100.0, 100.0 - (distance / 1.2))), 1)

    if student_id and distance <= threshold:
        annotated = draw_face_annotations(bgr, [bbox], f"{student_id} ({similarity_pct}%)", recognized=True)
        return {
            "status": "recognized",
            "recognized": True,
            "student_id": student_id,
            "distance": round(float(distance), 2),
            "confidence_pct": similarity_pct,
            "message": f"Matched student ID {student_id}. Please confirm before saving attendance.",
            "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        }

    annotated = draw_face_annotations(bgr, [bbox], "Unknown Face", recognized=False)
    return {
        "status": "unknown",
        "recognized": False,
        "distance": round(float(distance), 2),
        "confidence_pct": similarity_pct,
        "message": (
            f"Unknown face (LBPH distance {distance:.1f} exceeded threshold {threshold:.1f}). "
            "Use manual attendance backup or register this student first."
        ),
        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
    }
