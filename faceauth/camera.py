"""OpenCV webcam workflows used by the FaceAuth command-line application."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .core import AuthenticationResult, FaceAuthError, FaceAuthService, RegistrationResult


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise FaceAuthError(
            f"Could not open camera {camera_index}. Check that it is connected and not in use by another app."
        )
    return camera


def capture_registration_samples(
    service: FaceAuthService,
    name: str,
    count: int = 20,
    camera_index: int = 0,
) -> RegistrationResult:
    """Collect `count` single-face samples; press Q to cancel early."""
    if count < 5:
        raise FaceAuthError("Use at least 5 samples for a meaningful demo.")
    camera = _open_camera(camera_index)
    captured: list[np.ndarray] = []
    last_capture_at = 0.0
    try:
        while len(captured) < count:
            success, frame = camera.read()
            if not success:
                raise FaceAuthError("Camera returned an empty frame.")
            boxes = service.detect_faces(frame)
            preview = service.draw_detection_overlay(frame, boxes)
            instructions = f"Registering {name}: {len(captured)}/{count} | Slowly turn left/right | Q = cancel"
            cv2.putText(preview, instructions, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # A short delay avoids writing near-identical adjacent frames.
            now = time.monotonic()
            if len(boxes) == 1 and now - last_capture_at >= 0.30:
                captured.append(service.prepare_face(frame, boxes[0]))
                last_capture_at = now
                cv2.putText(preview, "Sample captured", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            elif len(boxes) > 1:
                cv2.putText(preview, "Only one face, please", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("FaceAuth - Registration", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise FaceAuthError("Registration cancelled. No samples were saved.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return service.register_faces(name, captured)


def authenticate_from_camera(
    service: FaceAuthService,
    camera_index: int = 0,
    stable_frames: int = 5,
) -> AuthenticationResult:
    """Scan continuously and decide after several consistent live frames."""
    if stable_frames < 1:
        raise FaceAuthError("stable_frames must be at least 1")
    camera = _open_camera(camera_index)
    last_result: AuthenticationResult | None = None
    candidate: tuple[str, str | None] | None = None
    candidate_frames = 0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise FaceAuthError("Camera returned an empty frame.")
            last_result = service.authenticate(frame)
            preview = service.draw_authentication_overlay(frame, last_result)
            decision_key = (last_result.status, last_result.name)
            if last_result.status in {"GRANTED", "DENIED"}:
                if decision_key == candidate:
                    candidate_frames += 1
                else:
                    candidate = decision_key
                    candidate_frames = 1
            else:
                candidate = None
                candidate_frames = 0

            cv2.putText(
                preview,
                f"Scanning automatically | Hold still: {candidate_frames}/{stable_frames} | Q = exit",
                (12, preview.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.imshow("FaceAuth - Authentication", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return AuthenticationResult("CANCELLED", "Authentication cancelled.")
            if candidate_frames >= stable_frames:
                # Keep the outcome readable before returning it to the terminal.
                preview = service.draw_authentication_overlay(frame, last_result)
                cv2.putText(preview, "Decision complete", (12, preview.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("FaceAuth - Authentication", preview)
                cv2.waitKey(1500)
                return last_result
    finally:
        camera.release()
        cv2.destroyAllWindows()
