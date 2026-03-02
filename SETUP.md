# ISL Word Interpreter - Setup Instructions

## Issue Fixed

Your code had a MediaPipe compatibility issue:
- **Original Error**: `AttributeError: module 'mediapipe' has no attribute 'solutions'`
- **Cause**: MediaPipe 0.10.x changed the API structure and removed the `solutions` module
- **Solution**: Updated the code to use the new MediaPipe `tasks.vision` API

## What Changed

1. **hand_detector.py** - Updated to use the new MediaPipe 0.10+ API
2. **requirements.txt** - Added version constraints for better compatibility
3. **download_model.py** - Created helper script to download the required model file

## Getting Started

### Step 1: Download the MediaPipe Model

The hand detection requires a model file (`hand_landmarker.task`) to function. Run:

```bash
python download_model.py
```

**If automatic download fails**, download manually:
1. Visit: https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task
2. Save the file as `hand_landmarker.task` in this directory

Or visit Google's MediaPipe Hand Landmarker page:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

### Step 2: Run the Application

Once the model file is in place:

```bash
python main.py
```

## Troubleshooting

### Model file not found error
- Make sure `hand_landmarker.task` is in the same directory as `main.py`
- Check file size is ~23 MB (the file should be fairly large)

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- For Python 3.13+, you need MediaPipe 0.10.0 or later

### Camera/Video issues
- Make sure your webcam is connected and not being used by another application
- Press 'q' to quit the application

## Dependencies

- **opencv-python** - For image processing and webcam access
- **mediapipe** - For hand detection and landmark tracking
- **numpy** - For numerical operations
- **pyttsx3** - For text-to-speech functionality

All dependencies are listed in `requirements.txt`.
