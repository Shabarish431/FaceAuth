# FaceAuth final report

**Presenter:** [name]  
**Date:** [date]  
**Environment:** Windows, Python [version], OpenCV [version], webcam [model]

## 1. Objective

Build an offline prototype that registers a face from a webcam and grants or denies access by comparing a new camera frame with locally stored examples.

## 2. Implementation

The system has four stages:

1. A Haar Cascade face detector locates faces in each OpenCV frame.
2. The selected face is grayscale-converted, contrast-normalized, and resized to 200 × 200 pixels.
3. During registration, multiple cropped samples are saved locally and an LBPH recognizer is trained.
4. During authentication, LBPH returns the closest label and a distance. A distance at or below [threshold] results in **Access Granted**; all other outcomes result in **Access Denied**.

The application rejects ambiguous frames with zero or multiple detected faces.

## 3. Test plan and results

| Scenario | Expected outcome | Observed outcome | Notes |
| --- | --- | --- | --- |
| Registered user, normal indoor lighting | Granted | [result] | [distance] |
| Unregistered user | Denied | [result] | [distance] |
| No face visible | Denied | [result] | |
| Two faces visible | Denied | [result] | |
| Registered user, lower lighting | [expected] | [result] | |
| Registered user, rotated / farther away | [expected] | [result] | |

## 4. Findings

[Summarize which conditions gave reliable matches, which produced false denials or false accepts, and whether the threshold was adjusted.]

## 5. Limitations and ethics

This is an educational demonstration rather than a secure authentication system. It does not perform liveness detection, so a replayed image may be accepted. Performance depends on lighting, pose, image quality, and the limited training set. Facial data is sensitive biometric data: subjects must consent, samples must stay local, and `data/` should be deleted after the demo unless retention has been agreed.

## 6. Future improvements

- Add liveness detection and anti-spoofing checks.
- Replace the classical recognizer with a well-evaluated embedding model.
- Encrypt data at rest and add consent, deletion, and audit workflows.
- Evaluate errors with a representative, consented test set before any real-world use.
