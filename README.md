# Hand-Gesture-Recognition

# Hand Gesture Recognition Project

This project uses **MediaPipe** and **TensorFlow/Keras** to recognize hand gestures for both numbers and predefined actions. It includes instructions for collecting data, training the model, and running real-time gesture recognition.

## Requirements

Ensure the following are installed:
- Python 3.8 or later
- A webcam for real-time gesture recognition
- Supported operating systems: Windows, Linux, or macOS

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <github repo>
   cd hand-gesture-recognition
2. **Create virtual Env**
   ```bash
   python -m venv venv
3. **Activate venv**
   ```bash
    source venv/bin/activate
4. **Install requirment**
   ```bash
   pip install -r requirements.txt
5. **Confirm Installation**
   ```bash
   python -c "import cv2, tensorflow, mediapipe; print('All libraries are installed!')"

**Collecting Data for Gesture Recognition**
Prepare the Dataset Folder

Create a DATASET folder in the root directory of your project.

Inside DATASET, create subfolders for each class (e.g., OK, THUMBS_UP, PALM_IN).

Example directory structure:

```bash
hand-gesture-recognition/
├── DATASET/
│   ├── OK/
│   ├── THUMBS_UP/
│   ├── PALM_IN/
│   ├── PALM_OUT/
│   ├── THUMBS_DOWN/
Run the Data Collection Script Use the provided script to collect images for each gesture:
```

```bash
  python getImage.py
 ```

The script will open your webcam.

Perform the gesture in front of the webcam while staying inside the ROI (Region of Interest).

Press q to stop capturing after collecting enough images for the gesture.

Repeat for All Gestures Run the script for each gesture and save the data in their respective folders.
