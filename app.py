import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="X-Ray Contraband Detector",
    page_icon="🧳",
    layout="centered"
)

st.title("🧳 X-Ray Security Scanner")
st.write("Upload a baggage X-ray image to detect potential contraband using our custom YOLOv8 model.")

# --- Load the Model ---
# The @st.cache_resource decorator ensures the model only loads once, 
# preventing lag every time the user interacts with the page.
@st.cache_resource
def load_model():
    # Update this path if you place your weights elsewhere
    return YOLO('best.pt') 
    #return YOLO('yolov8s.pt') #for test the UI

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model. Ensure 'best.pt' is in the same directory. Error: {e}")
    model_loaded = False

# --- UI Sidebar Settings ---
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- Main App Logic ---
if model_loaded:
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Scan")
            st.image(image, use_container_width=True)
            
        with st.spinner("Analyzing scan..."):
            # Run YOLO inference
            # We pass the PIL image directly. conf filters out weak predictions.
            results = model.predict(source=image, conf=confidence_threshold)
            
            # The results object contains a method to plot the bounding boxes
            # It returns a numpy array (BGR format), which we convert back to RGB for Streamlit
            res_plotted = results[0].plot()[:, :, ::-1] 
            
        with col2:
            st.subheader("Detection Results")
            st.image(res_plotted, use_container_width=True)
            
        # Display an expander with raw metrics
        with st.expander("Show Detection Details"):
            boxes = results[0].boxes
            if len(boxes) == 0:
                st.success("No contraband detected. Clear for transport.")
            else:
                st.warning(f"Detected {len(boxes)} potential threat(s).")
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    st.write(f"- **{class_name}**: {conf:.2%} confidence")
