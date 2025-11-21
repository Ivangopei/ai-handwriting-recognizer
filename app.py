import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Neural Digit Recognizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI POLISH ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 4rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('handwriting_model.keras')
    except:
        return None

model = load_model()

# --- SIDEBAR (YOUR PROFILE) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=80)
    st.title("About the Engineer")
    st.markdown("""
    **Name:** [Your Name Here]
    \n**Role:** Full Stack AI Engineer
    \n**Stack:** TensorFlow, Python, Streamlit
    """)
    st.divider()
    st.info("Draw a digit (0-9) on the canvas. The CNN model will predict the value in real-time.")
    st.write("---")
    st.caption("Powered by a Convolutional Neural Network trained on the MNIST dataset.")

# --- MAIN HEADER ---
st.title("🧠 AI Neural Network Recognizer")
st.markdown("<p style='text-align: center; color: #888;'>Handwritten Digit Classification System</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found. Please verify 'handwriting_model.keras' is in the directory.")
    st.stop()

# --- LAYOUT ---
col1, col2, col3 = st.columns([1, 2, 2])

with col2:
    st.subheader("Input Feed")
    st.caption("Draw a number (0-9) here:")
    
    # Canvas with a more professional look
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
    )

with col3:
    st.subheader("Real-Time Inference")
    
    if canvas_result.image_data is not None:
        # Preprocessing
        # ... (Start of processing block)
        img_data = canvas_result.image_data.astype('uint8')
        
        # Convert to grayscale
        img = Image.fromarray(img_data).convert('L')
        img_array = np.array(img)
        
        # --- SMART CROPPING LOGIC ---
        # 1. Find the bounding box of the drawing (where is the user's ink?)
        # We find all rows and columns that are NOT black
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        
        if np.sum(rows) > 0: # Only if they drew something
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop the image to just the number
            img_array = img_array[rmin:rmax+1, cmin:cmax+1]
            
            # Convert back to PIL to resize nicely
            img = Image.fromarray(img_array)
            
            # Resize to 20x20 (leaving 4px padding on all sides like MNIST)
            img = img.resize((20, 20))
            
            # Create a blank 28x28 black canvas
            new_img = Image.new('L', (28, 28), 0)
            
            # Paste the resized number into the center
            new_img.paste(img, (4, 4))
            
            # Use this new centered image for prediction
            img_array = np.array(new_img)
            img_array = img_array / 255.0
            img_final = img_array.reshape(1, 28, 28, 1)
            
            # ... (Continue with prediction as before)
            prediction = model.predict(img_final)
            # ...
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_final = img_array.reshape(1, 28, 28, 1)
        
        if np.sum(img_array) > 0:
            # Prediction
            prediction = model.predict(img_final)
            guessed_number = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Visual Output
            st.metric(label="Predicted Digit", value=str(guessed_number))
            
            st.write("Confidence Score:")
            st.progress(float(confidence))
            st.caption(f"Model is {confidence*100:.2f}% sure")
            
            # Detailed Chart
            st.write("Probability Distribution:")
            chart_data = pd.DataFrame(prediction[0], columns=["Probability"])
            st.bar_chart(chart_data)
            
        else:
            st.info("Waiting for input...")
            st.metric(label="Predicted Digit", value="-")

# --- TECHNICAL DETAILS EXPANDER (FOR RECRUITERS) ---
with st.expander("See How It Works (Technical Architecture)"):
    st.markdown("""
    ### Convolutional Neural Network (CNN) Architecture
    This model uses a sequential architecture optimized for computer vision tasks:
    * **Input Layer:** 28x28x1 Grayscale tensors.
    * **Conv2D Layers:** Feature extraction using 32/64 filters (ReLU activation).
    * **MaxPooling:** Spatial downsampling to reduce dimensionality.
    * **Dropout (0.2):** Regularization to prevent overfitting.
    * **Softmax Output:** Probability distribution across 10 digit classes.
    """)
