# FaceAuth

FaceAuth is a local, classroom-friendly face authentication demonstration. It detects one face with OpenCV's Haar cascade, compares it to locally registered samples using LBPH face recognition, and reports **Access Granted** or **Access Denied**.

> This is an educational prototype, not a security system. It has no liveness detection and should never protect money, personal data, or a physical location.

## What is included

- webcam registration of 20 face samples per user
- local model training and storage in `data/`
- webcam authentication with clear allow/deny feedback
- static-image face detection and authentication commands for a camera-free demo
- handling for no face, multiple faces, unknown faces, and an untrained model

## Setup

Use Python 3.10 or newer. From this folder, create a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or, on Windows, run the provided setup helper:

```powershell
.\setup.ps1
```

If PowerShell prevents activation, run `Set-ExecutionPolicy -Scope Process Bypass` once, then activate again. You may also run `.venv\Scripts\python.exe` directly instead of activating it.

## Fast demo

1. Register yourself. Keep your face centered; the app collects samples automatically. Move very slightly during capture so the samples include natural variation.

   ```powershell
   python app.py register --name Alice --samples 20
   ```

2. Verify that the local samples exist.

   ```powershell
   python app.py list-users
   ```

3. Authenticate. A live preview opens and scans automatically. Hold one face in view for five matching frames; the app displays the decision and closes automatically.

   ```powershell
   python app.py authenticate
   ```

4. For an unknown-face example, ask a different person to press Space. The result should be **Access Denied**.

`Q` or `Esc` exits either camera window. Pass `--camera 1` if an external webcam is camera 1.

## Static image commands

These commands make it easy to demonstrate the detection and matching stages with prepared photos:

```powershell
python app.py detect-image .\photo.jpg --output .\detected_faces.jpg
python app.py authenticate-image .\photo.jpg --output .\authentication_result.jpg
```

The output image has an annotation and is safe to show on a projector. The commands have non-zero exit codes for denied authentication; that is expected and makes them automation-friendly.

## Useful commands

```powershell
# More strict means fewer false accepts but more false denials. Start at 62.
python app.py --threshold 55 authenticate

# Require more matching live frames before the automatic decision.
python app.py authenticate --stable-frames 8

# View locally registered users
python app.py list-users

# Delete one user and rebuild the model
python app.py remove-user --name Alice
```

## How it works

1. **Detection:** pretrained Haar cascades find frontal and left/right profile face bounding boxes.
2. **Preprocessing:** the selected face is converted to grayscale, contrast-normalized, and resized to 200 × 200 pixels.
3. **Registration:** cropped examples are stored under `data/samples/<name>/` and an LBPH recognizer is trained from all stored examples.
4. **Authentication:** exactly one face must be present. LBPH returns a distance to the closest known user; distance at or below the threshold produces **Access Granted**, otherwise **Access Denied**.

The threshold is a tuning trade-off, not a probability. Lower values are stricter. Lighting, pose, distance, masks, camera quality, and the small amount of training data all affect performance. Profile detection helps with modest side turns, but this lightweight educational recognizer is not reliable for a fully turned, heavily tilted, or partially hidden face.

## Privacy and demo guidance

- Get consent before registering or showing anyone's face.
- `data/` holds biometric samples and a trained model. It is ignored by Git; delete it after the demo if it is no longer needed.
- Avoid presenting this as identity proof or access control. A printed/photo replay can fool it because it has no liveness check.
- Test the demo in the same lighting and camera position as the presentation. Register several samples per person and begin with one person at a time.

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| `LBPH recognition needs opencv-contrib-python` | Reinstall exactly from `requirements.txt`; do not install both `opencv-python` and `opencv-contrib-python` in the same environment. |
| Camera cannot open | Close Zoom/Teams/Camera and try `--camera 1`. Check Windows camera permissions. |
| No face detected | Improve front lighting, move closer, keep the face upright and unobstructed. |
| Known user is denied | Re-register 20–30 varied samples, then try a slightly higher threshold such as `--threshold 68`. |
| Unknown user is allowed | Use a lower threshold such as `--threshold 55`; this is a known limitation of a simple educational recognizer. |

## Project layout

```text
app.py                 command-line application
faceauth/core.py       detection, storage, training, decision logic
faceauth/camera.py     webcam registration and authentication views
requirements.txt       Python dependencies
setup.ps1              Windows setup helper
REPORT_TEMPLATE.md     editable final-project report
data/                  created at runtime; private local biometric data
```
