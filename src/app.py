import streamlit as st
from shutil import copyfile
from object_counting import process_video_and_count, process_image_and_count
import tempfile
import os

# Custom CSS for styling the layout and fonts
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1a2980, #26d0ce); /* Gradient background */
    }
    .stApp {
        background: linear-gradient(to right, #1a2980, #26d0ce); /* Gradient background */
    }
    header {
        background-color: orange;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
        margin-bottom: 20px;
        text-align: center;
    }
    .header-content {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header-content h1 {
        margin: 0;
    }
    .middle-section {
        padding: 10px;
        background-color: rgba(0,0,0,0.5);
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
        margin: 20px 0;
    }
    .highlight {
        font-weight: 700;
        font-size:900;
        color:orange;  /* Tomato color for highlighting */
            
    }
    .styled-text {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 30px;
        font-style: normal;
        font-weight: 800;
    }
    .st-radio>div, .st-multiselect>div, .st-file-uploader>div {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# App title and header
st.markdown("""
<header>
    <div class="header-content">
        <h1>Object Detection and Counting</h1>
    </div>
</header>
""", unsafe_allow_html=True)

# Option to select input type
st.markdown('<div class="middle-section">', unsafe_allow_html=True)
st.markdown('<p class="styled-text">Select the input type:</p>', unsafe_allow_html=True)
input_type = st.radio("Select input type:", ("Video", "Image"), label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# File uploader based on input type
st.markdown('<div class="middle-section">', unsafe_allow_html=True)
if input_type == "Video":
    st.markdown('<p class="styled-text">Upload a mp4 video...</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload video", type=["mp4"], label_visibility="collapsed")
    file_type = "video"
elif input_type == "Image":
    st.markdown('<p class="styled-text">Upload an image...</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"], label_visibility="collapsed")
    file_type = "image"
st.markdown('</div>', unsafe_allow_html=True)

# Selection of object classes
st.markdown('<div class="middle-section">', unsafe_allow_html=True)
st.markdown('<p class="styled-text">Select objects to count:</p>', unsafe_allow_html=True)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

selected_classes = st.multiselect(
    '',
    options=classNames,
    default=['cup', 'fork', 'spoon', 'knife', 'bed', 'tvmonitor', 'sofa', 'laptop', 'apple', 'sofa', 'stop sign']
)

# Convert class names to class IDs
class_ids = [classNames.index(cls) for cls in selected_classes if cls in classNames]
st.markdown('</div>', unsafe_allow_html=True)

# Confidence threshold slider
st.markdown('<div class="middle-section">', unsafe_allow_html=True)
st.markdown('<p class="styled-text">Adjust detection confidence threshold:</p>', unsafe_allow_html=True)
confidence_threshold = st.slider('', min_value=0.0, max_value=1.0, value=0.5)
st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded file and display results
if uploaded_file is not None and len(selected_classes) > 0:
    with st.spinner('Processing...'):
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.type.split("/")[-1]}')
        tfile.write(uploaded_file.getvalue())
        file_path = tfile.name

        # Process file based on type
        run_dir = "runs/temp"
        os.makedirs(run_dir, exist_ok=True)

        if file_type == "video":
            object_counts, output_video_path, input_video_path = process_video_and_count(file_path, 'yolov8s.pt', class_ids, run_dir, confidence_threshold)
            
            # Display input video
            st.markdown('<div class="middle-section">', unsafe_allow_html=True)
            st.subheader("Input Video")
            st.video(file_path)
            st.markdown('</div>', unsafe_allow_html=True)

            

        elif file_type == "image":
            object_counts, output_path = process_image_and_count(file_path, 'yolov8s.pt', class_ids, run_dir, confidence_threshold)
            # Display input image
            st.markdown('<div class="middle-section">', unsafe_allow_html=True)
            st.subheader("Input Image")
            st.image(file_path)
            st.markdown('</div>', unsafe_allow_html=True)

            # Display processed image
            st.markdown('<div class="middle-section">', unsafe_allow_html=True)
            if os.path.exists(output_path):
                st.subheader("Processed Image")
                st.image(output_path)
            else:
                st.error(f"Error: The image file was not found at {output_path}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Display object counts
        st.markdown('<div class="middle-section">', unsafe_allow_html=True)
        st.subheader("Object Counts")
        count_html = "<ol>"
        for i, (class_id, count) in enumerate(object_counts.items()):
            # Check if class_id is an integer or string
            if isinstance(class_id, int):
                class_name = classNames[class_id]
            else:
                class_name = class_id
            count_html += f"<li><span class='highlight'>{class_name}</span>: {count}</li>"
        count_html += "</ol>"
        st.markdown(count_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Option to download results
    st.markdown('<div class="middle-section">', unsafe_allow_html=True)
    st.markdown('<p class="styled-text">Download Results:</p>', unsafe_allow_html=True)
    if file_type == "video" and output_video_path:
        st.download_button(label="Download Processed Video", data=open(output_video_path, "rb"), file_name="processed_video.mp4")
    elif file_type == "image" and output_path:
        st.download_button(label="Download Processed Image", data=open(output_path, "rb"), file_name="processed_image.png")
    st.markdown('</div>', unsafe_allow_html=True)
