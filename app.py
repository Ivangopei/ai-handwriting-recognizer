# LIBRARIES
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# BROWSER TAB SETTINGS
st.set_page_config(
    page_title = "AI Digit Recognizer",
    page_icon = "🧠",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# WEB DESIGN CODE (CSS USED)
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {text-align: center; margin-bottom: 2rem;}
        .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
        div[data-testid="stMetricValue"] {font-size: 4rem;}
    </style>
""", unsafe_allow_html=True)

# LOADING AI BRAIN ("MODEL" VARIABLE CONTAINS THE AI BRAIN)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('handwriting_model.keras')
    except:
        return None

model = load_model()

# SIDEBAR (INFO ABOUT ME)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=80)
    st.title("About the Engineer")
    st.markdown("""
    **Name:** Ivan Gopei
    \n**Role:** AI Engineer
    \n**Stack:** TensorFlow, Python, Streamlit
    """)
    st.divider()
    st.info("Draw a digit (0-9). The AI uses a CNN with Data Augmentation to recognize rotated or messy inputs.")

# --- MAIN UI ---
st.title("AI Digit Recognizer")
st.markdown("<p style='text-align: center; color: #888;'>Handwritten Digit Classification System v2.0</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found. Please run train_advanced.py first!")
    st.stop()

col1, col2, col3 = st.columns([1, 2, 2])

with col2:
    st.subheader("Input Feed")
    st.caption("Draw a number (0-9) here:")
    
    canvas_result = st_canvas(
        fill_color = "rgba(255, 165, 0, 0.3)",
        stroke_width = 18,
        stroke_color = "#FFFFFF",
        background_color = "#000000",
        height = 300,
        width = 300,
        drawing_mode = "freedraw",
        key = "canvas",
        display_toolbar = True
    )

with col3:
    st.subheader("Real-Time Inference")
    
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data.astype('uint8')
        
        # --- SMART PREPROCESSING (ASPECT RATIO PRESERVED) ---
        img = Image.fromarray(img_data).convert('L')
        img_array = np.array(img)
        
        # 1. Find the bounding box
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        
        if np.sum(rows) > 0:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # 2. Crop the image to the content
            img_array = img_array[rmin:rmax+1, cmin:cmax+1]
            
            # 3. Calculate Aspect Ratio
            # We want to fit inside 20x20, but keep the shape!
            h, w = img_array.shape
            if h > w:
                # If tall (like a 1), scale height to 20, width proportionally
                factor = 20.0 / h
            else:
                # If wide, scale width to 20, height proportionally
                factor = 20.0 / w
            
            new_h = int(h * factor)
            new_w = int(w * factor)
            
            # Resize using the math above
            img = Image.fromarray(img_array)
            img = img.resize((new_w, new_h))
            
            # 4. Paste into center of 28x28 canvas
            new_img = Image.new('L', (28, 28), 0)
            
            # Math to find the center
            pad_left = int((28 - new_w) / 2)
            pad_top = int((28 - new_h) / 2)
            
            new_img.paste(img, (pad_left, pad_top))
            
            # 5. Predict
            img_array = np.array(new_img)
            img_array = img_array / 255.0
            img_final = img_array.reshape(1, 28, 28, 1)
            
            prediction = model.predict(img_final)
            guessed_number = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.metric(label="Predicted Digit", value=str(guessed_number))
            st.write("Confidence Score:")
            st.progress(float(confidence))
            st.caption(f"Model is {confidence*100:.2f}% sure")
            
            st.write("Probability Distribution:")
            chart_data = pd.DataFrame(prediction[0], columns=["Probability"])
            st.bar_chart(chart_data)
            
        else:
            st.info("Waiting for input...")







