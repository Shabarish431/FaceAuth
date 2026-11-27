from pathlib import Path

import cv2
import numpy as np
import pytest

from faceauth.core import FaceAuthError, FaceAuthService


@pytest.fixture()
def service(tmp_path: Path) -> FaceAuthService:
    return FaceAuthService(tmp_path / "data")


def test_no_face_is_denied(service: FaceAuthService) -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    result = service.authenticate(blank)
    assert result.status == "NO_FACE"
    assert not result.granted


def test_register_validates_a_name(service: FaceAuthService) -> None:
    with pytest.raises(FaceAuthError, match="Name"):
        service.register_faces("!!!", [np.zeros((200, 200), dtype=np.uint8)])


def test_registration_persists_samples_and_retrains(service: FaceAuthService) -> None:
    if not hasattr(cv2, "face"):
        pytest.skip("opencv-contrib-python is not installed")
    sample_a = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(sample_a, (100, 100), 50, 180, -1)
    sample_b = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(sample_b, (105, 100), 50, 180, -1)
    result = service.register_faces("Ada Lovelace", [sample_a, sample_b])
    assert result.name == "Ada_Lovelace"
    assert result.samples_added == 2
    assert service.model_path.exists()
    assert service.list_users() == {"Ada_Lovelace": 2}
