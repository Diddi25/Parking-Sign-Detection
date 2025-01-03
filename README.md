
# AI-Powered Parking Sign Translator

A cutting-edge solution for interpreting parking signs using deep learning and computer vision. This project simplifies parking regulations for drivers by translating complex parking signage into clear, actionable text.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)  
[![YOLOv7](https://img.shields.io/badge/YOLOv7-ObjectDetection-red)](https://github.com/WongKinYiu/yolov7)  
[![EasyOCR](https://img.shields.io/badge/EasyOCR-TextRecognition-green)](https://github.com/JaidedAI/EasyOCR)

---

## Overview

The **AI-Powered Parking Sign Translator** addresses the challenges drivers face when interpreting parking regulations from complex signage. This tool uses a combination of:

- **YOLOv7** for detecting relevant objects in parking signs.
- **EasyOCR** for extracting and interpreting text from detected signs.
- A **custom Python logic module** for understanding parking rules based on location, date, and time.

---

## Features

- **Object Detection**: Detect parking-related objects such as "P", time limits, fees, and more using YOLOv7.
- **Text Recognition**: Extract text from signs using EasyOCR for detailed analysis.
- **Rule Interpretation**: Provides concise parking instructions based on the current date and time.
- **Real-Time Results**: Outputs clear, actionable instructions to users.

---

## Installation

### Prerequisites

- Python 3.8+
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/parking-sign-translator.git
   cd parking-sign-translator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv7 weights:
   ```bash
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
   ```

4. Prepare dataset (if training a custom model):
   - Follow instructions in `data.yaml` for training and testing datasets.

---

## Usage

### Object Detection

Detect objects in parking signs:
```bash
python detect.py --weights yolov7.pt --source <image_or_video_path> --conf-thres 0.25
```

### Text Recognition

Extract and analyze text from parking signs:
```bash
python easyocr_detect.py
```

### Full Workflow

1. Run `detect.py` to identify objects in the image.
2. Use the bounding boxes to focus EasyOCR's text recognition.
3. Interpret parking rules using the custom logic module.

---

## Example Results

**Input**:  
![Input Sign](https://example.com/input-sign.png)

**Output**:
```plaintext
Based on the current date today: 2025-01-02, Thursday, at time: 10:30:
- You are allowed to park between: 08:00-18:00.
- You can park up to: 2 hours.
- A parking disc is required.
```

---

## Architecture

The system is divided into three key components:

1. **Object Detection**:
   - YOLOv7-based model trained on 15 custom parking-related classes.
   - Detects symbols, time frames, and text regions.

2. **Text Recognition**:
   - EasyOCR for extracting text from detected regions.
   - Processes multilingual text (English, Swedish, etc.).

3. **Rule Interpreter**:
   - Python logic interprets parking regulations based on detected objects and text.
   - Outputs clear parking instructions.

---

## Testing

Run tests on pre-labeled test images:
```bash
python test.py --weights yolov7.pt --data data.yaml
```

---

## Dataset

- **Training Data**: Collected and annotated parking signs from various urban and rural areas.
- **Labels**: Includes 15 classes, such as "weekday", "holiday", "P", and "parking fee".

Refer to `data.yaml` for dataset structure and label mapping.

---

## Contributions

This project was developed by:
- **Munira Ahmed** ([muniraah@kth.se](mailto:muniraah@kth.se))
- **Dipsikha Dutta** ([dipsikha@kth.se](mailto:dipsikha@kth.se))

---

## Acknowledgments

Special thanks to:
- [YOLOv7 Team](https://github.com/WongKinYiu/yolov7) for the object detection framework.
- [EasyOCR Team](https://github.com/JaidedAI/EasyOCR) for the text recognition library.

---

## License

This project is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
