# ai-handwriting-recognizer

A real-time computer vision application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and deployed via Streamlit Cloud.

### [Live Demo: Click Here to Try](https://ai-handwriting-recognizer-gvopm522ahmmulurjwv8xw.streamlit.app/)

## Tech Stack
* **Core:** Python 3.11
* **Deep Learning:** TensorFlow / Keras (CNN Architecture)
* **Computer Vision:** OpenCV logic (via NumPy/Pillow) for image preprocessing
* **Frontend:** Streamlit & HTML5 Canvas
* **Deployment:** Streamlit Cloud (CI/CD pipeline from GitHub)

## Key Features
* **Custom CNN Architecture:** Optimized with Conv2D, MaxPooling, and Dropout layers to achieve ~98% accuracy.
* **Smart Preprocessing:** Implemented a custom algorithm to center-crop and resize inputs while preserving aspect ratio (mimicking MNIST training data format).
* **Data Augmentation:** Model trained on rotated and zoomed images to handle "messy" real-world handwriting.
* **Mobile Responsive:** Fully touch-compatible interface.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Ivangopei/ai-handwriting-recognizer.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python -m streamlit run app.py
