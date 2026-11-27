"""Command-line entry point for the FaceAuth classroom demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from faceauth.camera import authenticate_from_camera, capture_registration_samples
from faceauth.core import DEFAULT_THRESHOLD, FaceAuthError, FaceAuthService


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FaceAuth: offline face registration, recognition, and access decisions."
    )
    parser.add_argument("--data-dir", default="data", help="Folder for local samples and model (default: data)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Maximum LBPH distance that grants access (default: 62)")
    subcommands = parser.add_subparsers(dest="command", required=True)

    register = subcommands.add_parser("register", help="Capture webcam samples for a user and train the model")
    register.add_argument("--name", required=True, help="User's display name (letters, digits, _ and - are retained)")
    register.add_argument("--samples", type=positive_int, default=20, help="Number of samples to capture (default: 20)")
    register.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    authenticate = subcommands.add_parser("authenticate", help="Continuously scan the webcam and authenticate automatically")
    authenticate.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    authenticate.add_argument("--stable-frames", type=positive_int, default=5, help="Matching frames required before deciding (default: 5)")

    detect = subcommands.add_parser("detect-image", help="Detect faces in an image and save an annotated copy")
    detect.add_argument("image", type=Path, help="Input image")
    detect.add_argument("--output", type=Path, default=Path("detected_faces.jpg"), help="Annotated output path")

    auth_image = subcommands.add_parser("authenticate-image", help="Authenticate a single face in an image")
    auth_image.add_argument("image", type=Path, help="Input image")
    auth_image.add_argument("--output", type=Path, default=Path("authentication_result.jpg"), help="Annotated output path")

    subcommands.add_parser("list-users", help="List local registered users and their sample counts")
    remove = subcommands.add_parser("remove-user", help="Remove one local user and retrain the model")
    remove.add_argument("--name", required=True, help="Name used at registration")
    return parser


def read_image(path: Path):
    if not path.is_file():
        raise FaceAuthError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise FaceAuthError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise FaceAuthError(f"Could not write image: {path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        service = FaceAuthService(args.data_dir, args.threshold)
        if args.command == "register":
            result = capture_registration_samples(service, args.name, args.samples, args.camera)
            print(f"Registered {result.name}: added {result.samples_added} samples (total: {result.total_samples}).")
        elif args.command == "authenticate":
            result = authenticate_from_camera(service, args.camera, args.stable_frames)
            print(result.message)
            if result.distance is not None:
                print(f"LBPH distance: {result.distance:.2f}; threshold: {service.threshold:.2f}")
            return 0 if result.granted else 1
        elif args.command == "detect-image":
            image = read_image(args.image)
            boxes = service.detect_faces(image)
            annotated = service.draw_detection_overlay(image, boxes, f"Faces detected: {len(boxes)}")
            save_image(args.output, annotated)
            print(f"Faces detected: {len(boxes)}. Annotated image saved to {args.output}")
        elif args.command == "authenticate-image":
            image = read_image(args.image)
            result = service.authenticate(image)
            save_image(args.output, service.draw_authentication_overlay(image, result))
            print(result.message)
            print(f"Annotated image saved to {args.output}")
            return 0 if result.granted else 1
        elif args.command == "list-users":
            users = service.list_users()
            if not users:
                print("No users registered yet.")
            else:
                for name, sample_count in users.items():
                    print(f"{name}: {sample_count} samples")
        elif args.command == "remove-user":
            if service.remove_user(args.name):
                print(f"Removed {args.name} and rebuilt the model.")
            else:
                print(f"No registered user named {args.name}.")
                return 1
        return 0
    except FaceAuthError as exc:
        print(f"FaceAuth error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
