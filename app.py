import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os # <-- IMPORTANT: Added for the robust file path

# ----------------------------------------------------
# Inject Custom CSS (The FIX for FileNotFoundError)
# ----------------------------------------------------
def local_css(file_name):
    """Function to read the CSS file using a robust path."""
    
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(current_dir, file_name)

    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Show an error if the CSS file is missing, but don't crash the app
        st.error(f"Error: CSS file '{file_name}' not found at {file_path}") 

# Call the function to load your styles
local_css("style.css")


# ----------------------------------------------------
# 1. CONFIGURATION (!! YOU MUST CHANGE THESE !!)
# ----------------------------------------------------
# A. The name of your saved model file
MODEL_PATH = 'resnet50_skin_model.h5' 

# B. The EXACT list of classes your model predicts, in the right order.
CLASS_NAMES = [
    'Actinic Keratoses (AK)', 
    'Basal Cell Carcinoma (BCC)', 
    'Benign Keratosis-like lesions (BKL)', 
    'Dermatofibroma (DF)', 
    'Melanoma (MEL)', 
    'Nevus (NV)', 
    'Vascular lesions (VASC)'
]
# C. The size your ResNet-50 model expects (usually 224x224 for ResNet)
IMAGE_SIZE = (224, 224) 


# ----------------------------------------------------
# 2. MODEL LOADING (Caching for efficiency)
# ----------------------------------------------------
@st.cache_resource 
def load_ml_model():
    """Tries to load the saved ResNet-50 model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load the model file at '{MODEL_PATH}'.")
        st.error(f"Please check the filename and path. Error details: {e}")
        st.stop() 

model = load_ml_model()

# ----------------------------------------------------
# 3. IMAGE PREPROCESSING FUNCTION
# ----------------------------------------------------
def preprocess_image(image_file):
    """Resizes and normalizes the image for the model."""
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image)
    img_array = img_array / 255.0 # Normalize 0-255 to 0-1
    return np.expand_dims(img_array, axis=0) # Add batch dimension

# ----------------------------------------------------
# 4. STREAMLIT APPLICATION UI AND LOGIC
# ----------------------------------------------------
st.title("🔬 Automated Skin Lesion Classifier")
st.subheader("Powered by ResNet-50 Deep Learning Model")
st.markdown("Upload a high-resolution image of a skin lesion for an instant classification.")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Image to Analyze', use_column_width=True)
    
    # Create a button to start the analysis
    if st.button('✨ Start Diagnosis'):
        
        with st.spinner('Analyzing image with ResNet-50... please wait...'):
            
            # --- 1. Preprocess & Predict ---
            processed_img = preprocess_image(uploaded_file)
            predictions = model.predict(processed_img)
            
            # --- 2. Get the final result ---
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]
            predicted_class = CLASS_NAMES[predicted_index]

            st.markdown("---")
            st.subheader("✅ Diagnosis Complete")
            
            # --- 3. Display Results with visual cues ---
            
            # Use color coding for serious classifications
            if predicted_class in ['Melanoma (MEL)', 'Basal Cell Carcinoma (BCC)']:
                st.error(f"🚨 **HIGH ALERT:** The model predicts **{predicted_class}**")
                st.write("This prediction suggests a potential malignancy. **Consult a specialist immediately.**")
            else:
                st.success(f"🟢 **LIKELY BENIGN:** The model predicts **{predicted_class}**")
                st.write("The model suggests this lesion is likely benign. **This is NOT medical advice.**")
            
            st.info(f"Model Confidence: **{confidence*100:.2f}%**")
            
            with st.expander("Show detailed class probabilities"):
                probability_data = {
                    "Class Name": CLASS_NAMES,
                    "Probability (%)": [f"{p*100:.2f}" for p in predictions[0]]
                }
                st.dataframe(probability_data)