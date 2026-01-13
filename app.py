import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM UI STYLING ---
# Manual CSS to clean up Streamlit's default spacing and center headers
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 0rem;}
        h1 {text-align: center; margin-bottom: 1.5rem;}
        .stButton>button {width: 100%; border-radius: 8px; height: 3em;}
        div[data-testid="stMetricValue"] {font-size: 3.5rem; color: #4CAF50;}
    </style>
""", unsafe_allow_html=True)

# --- MODEL UTILITIES ---
@st.cache_resource
def load_digit_model():
    """
    Loads the pre-trained CNN. Cached to prevent redundant 
    loading on every canvas interaction.
    """
    model_file = 'handwriting_model.keras'
    if not os.path.exists(model_file):
        return None
    try:
        # Loading the .keras format directly for better compatibility
        return tf.keras.models.load_model(model_file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Attempt to load the model early to catch setup issues
digit_classifier = load_digit_model()

# --- SIDEBAR: AUTHOR & INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=70)
    st.title("About the Engineer")
    st.markdown("""
    **Developer:** Ivan Gopei  
    **Project:** Digit Classification v2.0  
    **Tech Stack:** Python, TensorFlow, Streamlit  
    """)
    st.divider()
    st.info("""
    **How it works:** This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset. 
    The input is automatically centered and normalized before inference.
    """)

# --- MAIN INTERFACE ---
st.title("Handwritten Digit Recognizer")
st.markdown("<p style='text-align: center; color: #999;'>Real-time inference using a 2D CNN</p>", unsafe_allow_html=True)

# Fallback if the model is missing from the repo root
if digit_classifier is None:
    st.warning("⚠️ 'handwriting_model.keras' not found. Please ensure the model file is in your GitHub root.")
    st.stop()

# Layout: Canvas on the left, Predictions on the right
left_col, center_col, right_col = st.columns([1, 2, 2])

with center_col:
    st.subheader("Drawing Canvas")
    st.caption("Draw a single digit (0-9):")
    
    canvas_input = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="main_canvas",
        display_toolbar=True
    )

with right_col:
    st.subheader("AI Prediction")
    
    if canvas_input.image_data is not None and np.any(canvas_input.image_data):
        # Convert RGBA to Grayscale
        raw_pixels = canvas_input.image_data.astype('uint8')
        gray_image = Image.fromarray(raw_pixels).convert('L')
        pixel_array = np.array(gray_image)
        
        # --- PIPELINE: BOUNDING BOX & ASPECT RATIO ---
        # Find exactly where the drawing is to avoid center-bias errors
        rows = np.any(pixel_array, axis=1)
        cols = np.any(pixel_array, axis=0)
        
        if np.any(rows):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop to the content
            digit_region = pixel_array[rmin:rmax+1, cmin:cmax+1]
            
            # Preserve aspect ratio while fitting into a 20x20 box (MNIST style)
            h, w = digit_region.shape
            scale_ratio = 20.0 / max(h, w)
            new_dim = (int(w * scale_ratio), int(h * scale_ratio))
            
            rescaled_digit = Image.fromarray(digit_region).resize(new_dim)
            
            # Create the final 28x28 MNIST-standard frame
            mnist_frame = Image.new('L', (28, 28), 0)
            
            # Centering math using floor division
            x_offset = (28 - new_dim[0]) // 2
            y_offset = (28 - new_dim[1]) // 2
            mnist_frame.paste(rescaled_digit, (x_offset, y_offset))
            
            # --- INFERENCE ---
            # Normalize pixel values to [0, 1]
            input_tensor = np.array(mnist_frame) / 255.0
            input_batch = input_tensor.reshape(1, 28, 28, 1)
            
            raw_prediction = digit_classifier.predict(input_batch, verbose=0)
            predicted_class = np.argmax(raw_prediction)
            confidence_val = np.max(raw_prediction)
            
            # Display metrics and visual feedback
            st.metric(label="Predicted Digit", value=str(predicted_class))
            st.write(f"Confidence: **{confidence_val*100:.1f}%**")
            st.progress(float(confidence_val))
            
            # Probability chart for transparency
            prob_df = pd.DataFrame(raw_prediction[0], columns=["Probability"])
            st.bar_chart(prob_df)
            
    else:
        st.info("Start drawing to see real-time predictions.")
