
# AI-Powered Parking Sign Translator

An AI project for interpreting parking signs using deep learning and computer vision. This project simplifies parking regulations for drivers by translating parking signage into comprehensible and simple text.

---

## Contributions

This project was developed by:
- **Dipsikha (Diddi) Dutta** ([dipsikha@kth.se](mailto:dipsikha@kth.se))
- **Munira Ahmed** ([muniraah@kth.se](mailto:muniraah@kth.se))

---

## Usage

### Prerequisites

- Python 3.8-3.11
- Numpy 1.24
- Tensorflow
- PyTorch
- PyTorchVision
- EasyOCR
- NVIDIA CUDA Toolkit
- Compatible GPU-routines (we used NVIDIA GeForce GTX 1080)

### Installing steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/parking-sign-translator.git
   cd parking-sign-translator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   pip install easyocr
   ```

#### Check GPU and CUDA compatibility:
   ```bash
   py cuda_check.py
   ```

#### Using our custom dataset for testing:
   - Ask one of the contributors for API-key to 
   ```bash
   rf = Roboflow(api_key="fill in here")
   ```
   - in this file and run:
   ```bash
   py download_dataset.py
   ```

---

## How to run

### Object Detection

Detect objects in parking signs:
```bash
cd yolov7
python detect.py --weights yolov7.pt --source <image_or_video_path> --conf-thres 0.25
```

### Text Recognition

Extract and analyze text from parking signs:
```bash
cd ..
python end2end.py
```

---

## Example Results

**Output**:
```plaintext
Based on the current date today: 2025-01-02, Thursday, at time: 10:30:
- You are allowed to park between: 08:00-18:00.
- You can park up to: 2 hours.
- A parking disc is required.
```

---


