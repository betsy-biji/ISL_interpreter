# hand_detector.py

import cv2
import mediapipe as mp
from config import *
import os
import sys

class HandDetector:
    def __init__(self):
        """Initialize hand detector with MediaPipe 0.10+ new API"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Path to the model file
            model_path = os.path.join(
                os.path.dirname(__file__),
                'hand_landmarker.task'
            )
            
            # If model doesn't exist, provide helpful error message
            if not os.path.exists(model_path):
                print("\n" + "="*60)
                print("ERROR: hand_landmarker.task model file not found!")
                print("="*60)
                print("\nTo fix this, download the model file:")
                print("1. Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker")
                print("2. Download the 'hand_landmarker.task' file")
                print("3. Place it in the script directory:")
                print(f"   {model_path}")
                print("\nAlternatively, run: python download_model.py")
                print("="*60 + "\n")
                
                self.hand_landmarker = None
                self.mp_draw = None
                return
            
            # Create hand landmarker options
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=MAX_HANDS,
                min_hand_detection_confidence=DETECTION_CONFIDENCE,
                min_hand_presence_confidence=TRACKING_CONFIDENCE
            )
            
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            self.mp_draw = mp.tasks.vision.drawing_utils
            self.vision = vision
            
        except Exception as e:
            print(f"Error initializing HandDetector: {e}")
            self.hand_landmarker = None
            self.mp_draw = None

    def detect_hands(self, frame):
        """Detect hand landmarks in the frame"""
        
        # Return empty results if detection is not available
        class EmptyResults:
            def __init__(self):
                self.multi_hand_landmarks = []
        
        if self.hand_landmarker is None:
            return EmptyResults()
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.hand_landmarker.detect(mp_image)
            
            # Create backward-compatible results wrapper
            class ResultWrapper:
                def __init__(self, landmarks):
                    self.multi_hand_landmarks = landmarks if landmarks else []
            
            return ResultWrapper(results.hand_landmarks)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return EmptyResults()

    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on the frame"""
        if self.hand_landmarker is None or self.mp_draw is None:
            return
        
        try:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.vision.HandLandmarksConnections.HAND_CONNECTIONS
            )
        except Exception as e:
            print(f"Drawing error: {e}")