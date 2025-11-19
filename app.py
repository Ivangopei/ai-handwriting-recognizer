import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# 1. Load the Model (Cached so it doesn't reload every time you draw)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('handwriting_model.keras')

model = load_model()

# 2. App Layout - Make it look professional
st.set_page_config(page_title="Neural Digit Recognizer", layout="wide")

st.title("🧠 AI Made for Katie P (Pookie)")
st.markdown("Draw a digit (0-9) on the left. The AI will analyze pixels in real-time.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Feed")
    # Create the drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=20,
        stroke_color="#FFFFFF", # White pen
        background_color="#000000", # Black background (matching MNIST style)
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    st.caption("Draw using your mouse or trackpad.")

with col2:
    st.subheader("AI Analysis")
    
    if canvas_result.image_data is not None:
        # Get the image data from the canvas
        img_data = canvas_result.image_data.astype('uint8')
        
        # Convert to standard Image format and Grayscale
        img = Image.fromarray(img_data)
        img = img.convert('L') # Convert to grayscale
        
        # Resize to 28x28 (what the model expects)
        img = img.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize (0 to 1)
        img_array = img_array / 255.0
        
        # Reshape for the model (1 image, 28x28 pixels, 1 color channel)
        img_final = img_array.reshape(1, 28, 28, 1)
        
        # Only predict if the user has drawn something (sum of pixels > 0)
        if np.sum(img_array) > 0:
            prediction = model.predict(img_final)
            guessed_number = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Display Big Result
            st.metric(label="Predicted Digit", value=str(guessed_number), delta=f"{confidence*100:.1f}% Confidence")
            
            # Display Bar Chart of Probabilities
            st.write("### Probability Distribution")
            st.bar_chart(prediction[0])
        else:
            st.info("Waiting for input...")