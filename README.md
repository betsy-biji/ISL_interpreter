
ISL Word Interpreter
=====================

Brief
-----
Real-time Indian Sign Language (ISL) digit interpreter: captures webcam frames, detects hand landmarks using MediaPipe, computes finger open/closed status with geometric heuristics, maps gestures to digits (0–10), displays overlays, and speaks the detected digit.

Quick start
-----------
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Download MediaPipe hand landmarker model (if missing):

```bash
python download_model.py
```

- Run the interpreter:

```bash
python main.py
```

What it does (short)
--------------------
- Captures webcam frames with OpenCV.
- Uses MediaPipe `hand_landmarker.task` to detect up to two hands and return 21 landmarks per hand.
- Converts landmarks to a 5-bit finger-open/closed vector using wrist-relative distance heuristics (`gesture_classifier.py`).
- Maps single- or two-hand finger counts to digits; applies temporal smoothing to stabilize results.
- Renders landmarks and text on the video and uses TTS (`pyttsx3`) to speak detected digits.

Pipeline / Workflow (one slide)
-------------------------------
1. Capture frame (camera)
2. Detect hands & landmarks (MediaPipe via `hand_landmarker.task`)
3. Extract finger states (wrist-to-tip heuristics)
4. Classify to digit (single- or two-hand logic)
5. Smooth predictions (sliding window)
6. Output: overlay, FPS, spoken digit

Key files (one-line each)
-------------------------
- `main.py`: app entrypoint and real-time capture loop (smoothing, UI, TTS).
- `hand_detector.py`: MediaPipe hand landmarker integration and drawing helpers.
- `gesture_classifier.py`: convert 21 landmarks → finger state → numeric classification.
- `download_model.py`: helper to download `hand_landmarker.task` if needed.
- `config.py`: basic detection/tracking constants.
- `requirements.txt`: Python dependencies.

CGPI domain mapping (for PPT)
----------------------------
- Image/Object Identification & Processing: hand detection, landmark extraction, drawing overlays.
- Pose/Landmark-based Recognition: converting landmark sets into semantic finger states.
- Machine Learning (Model Integration): uses a pre-trained MediaPipe hand landmarker model (integration, not training).
- Signal Processing / Filtering: temporal smoothing of predictions to reduce noise.
- Human–Computer Interaction (HCI): real-time visual + spoken feedback for accessibility.
- Software Engineering: modular design, dependency management, asset download helper.

PPT-ready bullets (copy into slides)
----------------------------------
- Title: ISL Word Interpreter — real-time hand gesture to digit translator
- One-line purpose: Translate ISL digit gestures from webcam into visual and spoken outputs.
- Core pipeline: Capture → Detect (MediaPipe) → Extract landmarks → Compute finger states → Classify → Smooth → Display & Speak.
- Tech stack: Python, OpenCV, MediaPipe (hand_landmarker.task), NumPy, pyttsx3.
- Model: Uses MediaPipe `hand_landmarker.task` for accurate 21-point hand landmarks.
- Algorithms: geometric wrist-to-tip heuristics for finger detection; sliding-window smoothing for stability.
- Extensions: collect dataset & train classifier for words/phrases, add language options, improve robustness (illumination, occlusion).

Notes & next steps
------------------
- `train_model.py` and `create_synthetic_dataset.py` are placeholders for extending to a trainable classifier. Consider collecting labeled landmark sequences and training a sequence model (LSTM/Transformer) for word-level recognition.
- If you want, I can generate slide text for 4–6 PPT slides based on these bullets, or produce speaker notes.

License / Attribution
---------------------
This project integrates the MediaPipe hand landmarker model (download separately) and uses open-source Python libraries listed in `requirements.txt`.
