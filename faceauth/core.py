"""Face detection, local training, and authentication logic for FaceAuth.

This project deliberately uses OpenCV's Haar cascade and LBPH recognizer.  They
run entirely on-device, require no account or network connection, and make the
matching threshold easy to explain during a demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
from typing import Iterable

import cv2
import numpy as np


FACE_SIZE = (200, 200)
DEFAULT_THRESHOLD = 62.0
LABELS_FILE = "labels.json"
MODEL_FILE = "lbph_model.yml"


class FaceAuthError(RuntimeError):
    """A recoverable error caused by invalid data or a missing OpenCV feature."""


@dataclass(frozen=True)
class AuthenticationResult:
    """The decision made for one image or webcam frame."""

    status: str
    message: str
    name: str | None = None
    distance: float | None = None
    face_count: int = 0
    box: tuple[int, int, int, int] | None = None

    @property
    def granted(self) -> bool:
        return self.status == "GRANTED"


@dataclass(frozen=True)
class RegistrationResult:
    name: str
    samples_added: int
    total_samples: int


class FaceAuthService:
    """Owns the files and operations for one local FaceAuth data store."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        threshold: float = DEFAULT_THRESHOLD,
        face_size: tuple[int, int] = FACE_SIZE,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero")

        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
        self.model_path = self.data_dir / MODEL_FILE
        self.labels_path = self.data_dir / LABELS_FILE
        self.threshold = float(threshold)
        self.face_size = face_size
        self._loaded_model: object | None = None
        self._loaded_model_mtime_ns: int | None = None
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        cascade_dir = Path(cv2.data.haarcascades)
        frontal_path = cascade_dir / "haarcascade_frontalface_default.xml"
        profile_path = cascade_dir / "haarcascade_profileface.xml"
        self.detector = cv2.CascadeClassifier(str(frontal_path))
        self.profile_detector = cv2.CascadeClassifier(str(profile_path))
        if self.detector.empty():
            raise FaceAuthError(f"Could not load frontal-face Haar cascade at {frontal_path}")
        if self.profile_detector.empty():
            raise FaceAuthError(f"Could not load profile-face Haar cascade at {profile_path}")

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
        if not cleaned:
            raise FaceAuthError("Name must contain a letter or number.")
        return cleaned

    def _load_labels(self) -> dict[int, str]:
        if not self.labels_path.exists():
            return {}
        try:
            raw = json.loads(self.labels_path.read_text(encoding="utf-8"))
            return {int(key): str(value) for key, value in raw.get("label_to_name", {}).items()}
        except (OSError, ValueError, TypeError) as exc:
            raise FaceAuthError(f"Could not read {self.labels_path}: {exc}") from exc

    def _save_labels(self, labels: dict[int, str]) -> None:
        payload = {"label_to_name": {str(key): value for key, value in labels.items()}}
        self.labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _recognizer() -> cv2.face_LBPHFaceRecognizer:  # type: ignore[name-defined]
        face_module = getattr(cv2, "face", None)
        factory = getattr(face_module, "LBPHFaceRecognizer_create", None)
        if factory is None:
            raise FaceAuthError(
                "LBPH recognition needs opencv-contrib-python. Install dependencies "
                "with: python -m pip install -r requirements.txt"
            )
        return factory()

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return frontal and left/right profile face rectangles.

        The profile cascade detects one direction. Running it on the horizontally
        flipped frame detects the opposite direction as well, so registration can
        include modest left and right head turns.
        """
        if image is None or image.size == 0:
            return []
        gray = self._to_gray(image)
        settings = {"scaleFactor": 1.15, "minNeighbors": 5, "minSize": (70, 70)}
        frontal = self.detector.detectMultiScale(gray, **settings)
        right_profile = self.profile_detector.detectMultiScale(gray, **settings)
        flipped = cv2.flip(gray, 1)
        left_profile_flipped = self.profile_detector.detectMultiScale(flipped, **settings)
        image_width = gray.shape[1]
        left_profile = [
            (image_width - int(x) - int(width), int(y), int(width), int(height))
            for x, y, width, height in left_profile_flipped
        ]
        boxes = [tuple(int(value) for value in box) for box in frontal]
        boxes.extend(tuple(int(value) for value in box) for box in right_profile)
        boxes.extend(left_profile)
        return self._deduplicate_boxes(boxes)

    @staticmethod
    def _deduplicate_boxes(
        boxes: Iterable[tuple[int, int, int, int]],
        overlap_threshold: float = 0.35,
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping cascade detections of the same physical face."""
        kept: list[tuple[int, int, int, int]] = []
        for candidate in sorted(boxes, key=lambda box: box[2] * box[3], reverse=True):
            x, y, width, height = candidate
            candidate_area = width * height
            overlaps_existing = False
            for other_x, other_y, other_width, other_height in kept:
                intersection_width = max(0, min(x + width, other_x + other_width) - max(x, other_x))
                intersection_height = max(0, min(y + height, other_y + other_height) - max(y, other_y))
                intersection = intersection_width * intersection_height
                union = candidate_area + (other_width * other_height) - intersection
                if union and intersection / union >= overlap_threshold:
                    overlaps_existing = True
                    break
            if not overlaps_existing:
                kept.append(candidate)
        return kept

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise FaceAuthError("Expected a grayscale, BGR, or BGRA image.")
        return cv2.equalizeHist(gray)

    def prepare_face(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = box
        gray = self._to_gray(image)
        crop = gray[y : y + height, x : x + width]
        if crop.size == 0:
            raise FaceAuthError("Detected face crop was empty.")
        return cv2.resize(crop, self.face_size, interpolation=cv2.INTER_AREA)

    def register_faces(self, name: str, face_images: Iterable[np.ndarray]) -> RegistrationResult:
        """Persist already-cropped face samples for a user and train a new model."""
        safe_name = self._safe_name(name)
        faces = [cv2.resize(self._to_gray(face), self.face_size) for face in face_images if face is not None and face.size]
        if not faces:
            raise FaceAuthError("No usable face samples were captured.")

        user_dir = self.samples_dir / safe_name
        user_dir.mkdir(parents=True, exist_ok=True)
        start_index = len(list(user_dir.glob("*.png")))
        for offset, face in enumerate(faces, start=1):
            destination = user_dir / f"sample_{start_index + offset:03d}.png"
            if not cv2.imwrite(str(destination), face):
                raise FaceAuthError(f"Could not save sample: {destination}")

        self.train()
        return RegistrationResult(name=safe_name, samples_added=len(faces), total_samples=start_index + len(faces))

    def train(self) -> None:
        """Rebuild the LBPH model from all local samples."""
        user_dirs = sorted(path for path in self.samples_dir.iterdir() if path.is_dir())
        if not user_dirs:
            raise FaceAuthError("No registered users found. Register a user first.")

        images: list[np.ndarray] = []
        numeric_labels: list[int] = []
        label_to_name: dict[int, str] = {}
        for label, user_dir in enumerate(user_dirs):
            sample_paths = sorted(user_dir.glob("*.png"))
            usable = 0
            for sample_path in sample_paths:
                sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                if sample is None:
                    continue
                images.append(cv2.resize(sample, self.face_size))
                numeric_labels.append(label)
                usable += 1
            if usable:
                label_to_name[label] = user_dir.name

        if not images:
            raise FaceAuthError("No readable training images found.")

        recognizer = self._recognizer()
        recognizer.train(images, np.asarray(numeric_labels, dtype=np.int32))
        recognizer.write(str(self.model_path))
        self._save_labels(label_to_name)
        self._loaded_model = recognizer
        self._loaded_model_mtime_ns = self.model_path.stat().st_mtime_ns

    def _load_recognizer(self):
        """Load the model once and reuse it for smooth live camera scanning."""
        if not self.model_path.exists():
            raise FaceAuthError("No registered face model. Register a user first.")
        current_mtime = self.model_path.stat().st_mtime_ns
        if self._loaded_model is None or self._loaded_model_mtime_ns != current_mtime:
            recognizer = self._recognizer()
            recognizer.read(str(self.model_path))
            self._loaded_model = recognizer
            self._loaded_model_mtime_ns = current_mtime
        return self._loaded_model

    def list_users(self) -> dict[str, int]:
        """Return each registered user and the number of stored training samples."""
        result: dict[str, int] = {}
        for directory in sorted(path for path in self.samples_dir.iterdir() if path.is_dir()):
            result[directory.name] = len(list(directory.glob("*.png")))
        return result

    def remove_user(self, name: str) -> bool:
        """Remove one user's samples, then retrain (or clear the model)."""
        user_dir = self.samples_dir / self._safe_name(name)
        if not user_dir.exists():
            return False
        shutil.rmtree(user_dir)
        if self.list_users():
            self.train()
        else:
            self.model_path.unlink(missing_ok=True)
            self.labels_path.unlink(missing_ok=True)
            self._loaded_model = None
            self._loaded_model_mtime_ns = None
        return True

    def authenticate(self, image: np.ndarray) -> AuthenticationResult:
        """Authenticate an image only when it contains exactly one face."""
        boxes = self.detect_faces(image)
        if not boxes:
            return AuthenticationResult("NO_FACE", "Access denied: no face detected.")
        if len(boxes) > 1:
            return AuthenticationResult(
                "MULTIPLE_FACES",
                "Access denied: show exactly one face to the camera.",
                face_count=len(boxes),
            )
        if not self.model_path.exists():
            return AuthenticationResult(
                "NO_MODEL",
                "Access denied: no registered face model. Register a user first.",
                face_count=1,
                box=boxes[0],
            )

        labels = self._load_labels()
        if not labels:
            return AuthenticationResult("NO_MODEL", "Access denied: labels are missing.", face_count=1, box=boxes[0])
        try:
            recognizer = self._load_recognizer()
            label, distance = recognizer.predict(self.prepare_face(image, boxes[0]))
        except cv2.error as exc:
            raise FaceAuthError(f"Could not run recognition: {exc}") from exc

        name = labels.get(int(label))
        if name is not None and float(distance) <= self.threshold:
            return AuthenticationResult(
                "GRANTED",
                f"Access granted: {name} recognized.",
                name=name,
                distance=float(distance),
                face_count=1,
                box=boxes[0],
            )
        return AuthenticationResult(
            "DENIED",
            "Access denied: face is unknown.",
            distance=float(distance),
            face_count=1,
            box=boxes[0],
        )

    @staticmethod
    def draw_detection_overlay(
        image: np.ndarray,
        boxes: Iterable[tuple[int, int, int, int]],
        caption: str | None = None,
    ) -> np.ndarray:
        output = image.copy()
        for x, y, width, height in boxes:
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 210, 255), 2)
        if caption:
            cv2.putText(output, caption, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
        return output

    def draw_authentication_overlay(self, image: np.ndarray, result: AuthenticationResult) -> np.ndarray:
        output = image.copy()
        if result.box:
            x, y, width, height = result.box
            colour = (0, 200, 0) if result.granted else (0, 0, 220)
            cv2.rectangle(output, (x, y), (x + width, y + height), colour, 3)
        colour = (0, 200, 0) if result.granted else (0, 0, 220)
        cv2.putText(output, result.status.replace("_", " "), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 3)
        cv2.putText(output, result.message, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)
        if result.distance is not None:
            cv2.putText(output, f"LBPH distance: {result.distance:.1f} (threshold {self.threshold:.1f})", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        return output
