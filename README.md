
# AI-Powered Parking Sign Translator

An AI project for interpreting parking signs using deep learning and computer vision. This project simplifies parking regulations for drivers by translating parking signage into comprehensible and simple text.

---

## Contributors

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
- NVIDIA cuDNN
- Compatible GPU-routines (we used NVIDIA GeForce GTX 1080, CUDA v.11.8)

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
If something goes wrong, check cuda version:
   ```bash
   nvcc --version 
   ```
Install compatible torch dependency for your cuda version:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
   ```
   ```bash
   pip install torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
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
py detect.py --weights runs/train/exp17/weights/best.pt --conf 0.1 --source 'path to this image: p-skiva-skylt_jpg.rf.9cf6eba217156be500efb441efd91d90.jpg' --save-txt
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
Based on the current date today: 2025-01-03, Friday, at time: 18:07:
You are allowed to park between: 9-19.
You can park up to: 1 timmar.
You have to have a parking disc.

Övrig information:
Utanför tidsramen 9-19 kan du parkera längre än 1 timmar.
Utanför tidsramen 9-19 behöver du inte ha parkeringsskiva.
```

---


