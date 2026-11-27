# FaceAuth demonstration notes

## Suggested 5-minute flow

1. Explain that a camera frame is a NumPy array; OpenCV reads it frame by frame.
2. Run `python app.py register --name DemoUser --samples 20` and point out that only one face may be captured.
3. Run `python app.py list-users` to show the locally stored training set.
4. Run `python app.py authenticate`; the camera scans automatically once the face remains stable for five matching frames. Explain the granted decision and distance.
5. Repeat with an unregistered volunteer (or no face / two people) to show denial and edge-case handling.

## Talking points

- **Detection is not recognition.** Frontal and profile Haar cascades say where a likely face is. LBPH compares texture patterns in the cropped face to enrolled samples.
- **Distance is a similarity measure.** A lower LBPH distance is more similar. The configurable threshold turns it into an allow/deny decision.
- **Why exactly one face?** Authentication must make a decision about one claimant. Multiple faces make that ambiguous.
- **Why this is a prototype:** it does not detect a live person, cannot reliably identify people in hard conditions, and may perform unevenly across people and environments.

## Before mentees arrive

- Test the webcam in the presentation room and close apps that might reserve it.
- Register one presenter under the final lighting conditions.
- Prepare one consented photo for `detect-image` in case the webcam is unavailable.
- Do not retain `data/` after the session unless participants explicitly agree.
