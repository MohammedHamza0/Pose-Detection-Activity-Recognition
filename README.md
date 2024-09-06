# Pose Detection & Activity Recognition

This project demonstrates real-time human pose detection and activity recognition using YOLOv8 and OpenCV. The goal is to classify activities based on body posture by calculating angles between key body joints.

## 📦 **Features**

- **Pose Estimation**: Detects human body keypoints from video input using YOLOv8.
- **Angle Calculation**: Computes the angle between shoulder, hip, and knee to determine posture.
- **Activity Classification**: Differentiates between "Sitting" and "Standing" based on the angle.
- **Real-Time Visualization**: Displays detected poses, angles, and activity status on video frames.

## 🛠️ **Technologies Used**

- **YOLOv8**: For pose estimation and keypoint detection.
- **OpenCV**: For video processing and visualization.
- **Python**: For angle calculation and activity recognition.

## 🚀 **Getting Started**

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- [YOLOv8 model weights](https://github.com/ultralytics/ultralytics) (`yolov8s-pose.pt`)
- OpenCV library

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/MohammedHamza0/pose-detection-activity-recognition.git
    cd pose-detection-activity-recognition
    ```


### Usage

1. **Set the working directory:**

    Update the `os.chdir()` path in `HumanActivity.py` to point to your video file location.

2. **Run the script:**

    ```bash
    python HumanActivity.py
    ```

3. **View results:**

    The script will open a window displaying the video with real-time pose detection and activity classification. Press 'x' to exit the video window.

