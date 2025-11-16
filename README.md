# 🚗 Real-Time Lane Detection System

A computer vision-based lane detection system that processes video files and provides real-time webcam lane detection using OpenCV and Streamlit.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **Video Processing**: Upload and process video files with lane detection
- **Real-time Webcam Detection**: Live lane detection from your webcam
- **Lane Curvature Calculation**: Measures the radius of curvature of detected lanes
- **Vehicle Offset Measurement**: Calculates vehicle position relative to lane center
- **Download Processed Videos**: Export processed videos with lane overlays
- **Interactive UI**: Clean and intuitive Streamlit interface

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection mode)
- pip package manager

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/lane-detection-system.git
cd lane-detection-system
```

2. **Create a virtual environment (recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Video Upload Mode

1. Select "Upload Video" from the sidebar
2. Click "Browse files" and select a video file (MP4, AVI, or MOV)
3. Click "Process Video" to start detection
4. Download the processed video using the download button

### Using Webcam Mode

1. Select "Webcam" from the sidebar
2. Click "Start Webcam" to begin real-time detection
3. View live lane detection with curvature and offset metrics
4. Click "Stop Webcam" to end the session

## 🛠️ Technology Stack

- **OpenCV**: Computer vision operations and video processing
- **NumPy**: Numerical computations and array operations
- **Streamlit**: Web application framework
- **Python**: Core programming language

## 📊 Detection Algorithm

The lane detection system uses the following pipeline:

1. **Preprocessing**: Convert frame to grayscale and apply Gaussian blur
2. **Edge Detection**: Canny edge detection to identify lane boundaries
3. **Region of Interest**: Focus on the relevant road area
4. **Line Detection**: Hough Transform to detect lane lines
5. **Line Separation**: Classify lines into left and right lanes
6. **Polynomial Fitting**: Fit 2nd order polynomials to lane lines
7. **Smoothing**: Temporal smoothing for stable detection
8. **Overlay**: Draw detected lanes on the original frame
9. **Metrics**: Calculate curvature and vehicle offset

## 📁 Project Structure

```
lane-detection-system/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # License file
├── .gitignore            # Git ignore file
│
├── sample_videos/        # Sample test videos (optional)
│   └── test_video.mp4
│
└── docs/                 # Additional documentation (optional)
    ├── algorithm.md
    └── screenshots/
```

## 🎯 Key Parameters

You can adjust these parameters in the `LaneDetector` class:

- **Canny Edge Detection**: `low_threshold=50`, `high_threshold=150`
- **Hough Transform**: `threshold=50`, `minLineLength=50`, `maxLineGap=150`
- **Slope Filtering**: Left lane `< -0.5`, Right lane `> 0.5`
- **Smoothing Factor**: `0.7 * previous + 0.3 * current`

## 🐛 Troubleshooting

### Webcam not working
- Ensure your webcam is not being used by another application
- Check webcam permissions in your system settings
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

### Video processing slow
- Reduce video resolution before uploading
- Close other resource-intensive applications
- Consider using a smaller region of interest

### Lane detection inaccurate
- Adjust Canny edge detection thresholds
- Modify Hough Transform parameters
- Ensure good lighting conditions in video/webcam feed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV community for excellent computer vision tools
- Streamlit team for the amazing web framework
- Inspiration from autonomous driving research

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/lane-detection-system](https://github.com/yourusername/lane-detection-system)

## 📸 Screenshots

### Video Processing Mode
![Video Processing](docs/screenshots/video_mode.png)

### Webcam Mode
![Webcam Mode](docs/screenshots/webcam_mode.png)

### Detected Lanes
![Lane Detection](docs/screenshots/detection.png)

---

⭐ If you find this project useful, please consider giving it a star!
