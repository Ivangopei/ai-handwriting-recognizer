# AI Handwriting Recognizer
An application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and deployed through Streamlit Cloud.

# Why I Built This?
I always wondered how computers see our messy handwritings and could tell what exactly we wrote... 

By working on this project I could finally see what gives an "eyes" to AI, and how does AI's brain functions and how it learns based on the thousands of different handwritings. Finally, I wanted to show people my results - so, I connected my GitHub account with Streamlit and deployed the Digit Recognizer to the cloud. 

For people reading this, I want to give a fun challenge - confuse this AI by drawing a digit (0 - 9) in the most confusing way.


### [Live Demo: Click Here to Try](https://ai-handwriting-recognizer-gvopm522ahmmulurjwv8xw.streamlit.app/)

## How The Whole AI Functions (The Tech Stack)
* **The Brain (Architecture):** A Convolutional Neural Network built with TensorFlow/Keras featuring Conv2D, MaxPooling, and Dropout layers. I was able to achieve ~98% accuracy feeding data into AI.
* **The Interface:** Used Streamlit's interface and added a custom HTML5 canvas for smooth drawing.
* **The Glue:** OpenCV & Pillow handle the heavy lifting of cropping, resizing, and grayscale conversion to match the 28x28 pixel format the model expects.

## Key Features
* **Mobile Friendly:** Draw with your mouse on a laptop or your finger on a smartphone.
* **Reliable:** Achieved ~98% accuracy on recognizing digits.
* **Data Diversity:** AI was trained on the tens of thousands pieces data of the handwritten digits of the people across the world. 
   - **NOTE:** I accounted for regional differences, like how a "7" or a "1" might be written differently in Europe vs the US.
* **Robust to Messy Input:** Model trained on rotated and zoomed images to handle confusing real-world handwritings.


## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Ivangopei/ai-handwriting-recognizer.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   python -m streamlit run app.py


If you did everything correctly you should see this interface:
<img width="1501" height="813" alt="image" src="https://github.com/user-attachments/assets/f060776c-ca35-4369-91d4-64c35bfb525e" />
