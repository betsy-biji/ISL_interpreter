#!/usr/bin/env python3
"""
Download the MediaPipe hand_landmarker.task model file.

The hand_landmarker.task file is required for the ISL Word Interpreter to function.
This script will attempt to download it from the official MediaPipe repository.
"""

import os
import sys
import urllib.request
import urllib.error
import shutil

def download_with_progress(url, filename):
    """Download file with progress indication"""
    try:
        print(f"Downloading from: {url}")
        # Use urlretrieve with reporthook for progress
        def reporthook(blocknum, blocksize, totalsize):
            downloaded = blocknum * blocksize
            if totalsize > 0:
                percent = min(downloaded * 100 // totalsize, 100)
                print(f"\rProgress: {percent}% ({downloaded}/{totalsize} bytes)", end='')
        
        urllib.request.urlretrieve(url, filename, reporthook=reporthook)
        print(f"\n✓ Downloaded successfully: {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def main():
    print("="*70)
    print("MediaPipe Hand Landmarker Model Downloader")
    print("="*70)
    
    model_filename = 'hand_landmarker.task'
    
    # Check if already exists
    if os.path.exists(model_filename):
        file_size = os.path.getsize(model_filename)
        print(f"\n✓ Model file already exists: {model_filename} ({file_size} bytes)")
        return True
    
    # URLs to try (in order of preference)
    urls = [
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
    ]
    
    print(f"\nTarget file: {model_filename}")
    print(f"Size: ~23 MB\n")
    
    for url in urls:
        if download_with_progress(url, model_filename):
            print("\n✓ Model downloaded successfully!")
            print(f"You can now run: python main.py")
            return True
    
    # If all downloads fail, provide manual instructions
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("""
If automatic download failed, please download manually:

1. Visit: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
2. Look for the model download section 
3. Download the 'hand_landmarker.task' file (~23 MB tflite model)
4. Place it in the current directory:
   """ + os.getcwd() + f"""

Alternatively, try downloading from:
   https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task

After downloading, try running main.py again.
""")
    print("="*70)
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

