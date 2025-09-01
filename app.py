# import cv2
# import streamlit as st
# from ultralytics import YOLO

# def app():
#     st.header('Object Detection Web App')
#     st.subheader('Powered by YOLOv8')
#     st.write('Welcome!')
#     model = YOLO('yolov8n.pt')
#     object_names = list(model.names.values())

#     with st.form("my_form"):
#         uploaded_file = st.file_uploader("Upload video", type=['mp4'])
#         selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person']) 
#         min_confidence = st.slider('Confidence score', 0.0, 1.0)
#         st.form_submit_button(label='Submit')
            
#     if uploaded_file is not None: 
#         input_path = uploaded_file.name
#         file_binary = uploaded_file.read()
#         with open(input_path, "wb") as temp_file:
#             temp_file.write(file_binary)
#         video_stream = cv2.VideoCapture(input_path)
#         width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
#         height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
#         fourcc = cv2.VideoWriter_fourcc(*'h264') 
#         fps = int(video_stream.get(cv2.CAP_PROP_FPS)) 
#         output_path = input_path.split('.')[0] + '_output.mp4' 
#         out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height)) 

#         with st.spinner('Processing video...'): 
#             while True:
#                 ret, frame = video_stream.read()
#                 if not ret:
#                     break
#                 result = model(frame)
#                 for detection in result[0].boxes.data:
#                     x0, y0 = (int(detection[0]), int(detection[1]))
#                     x1, y1 = (int(detection[2]), int(detection[3]))
#                     score = round(float(detection[4]), 2)
#                     cls = int(detection[5])
#                     object_name =  model.names[cls]
#                     label = f'{object_name} {score}'

#                     if model.names[cls] in selected_objects and score > min_confidence:
#                         cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
#                         cv2.putText(frame, label, (x0, y0 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
#                 detections = result[0].verbose()
#                 cv2.putText(frame, detections, (10, 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
#                 out_video.write(frame) 
#             video_stream.release()
#             out_video.release()
#         st.video(output_path)

# if __name__ == "__main__":
#     app()
import cv2
import streamlit as st
from ultralytics import YOLO
import os
import shutil
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Object Detection Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    """Load custom CSS with Claude.ai inspired creamy theme"""
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styling with Claude.ai theme colors */
    .stApp {
        background: linear-gradient(135deg, #fef7ed 0%, #fed7aa 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    .main {
        background: transparent !important;
        color: #1f2937 !important;
    }
    
    /* Typography */
    .stMarkdown, .stText, p, span, div, label {
        color: #1f2937 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stFileUploader label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    .stRadio label, .stCheckbox label {
        color: #374151 !important;
        font-weight: 500 !important;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #ea580c 0%, #dc2626 50%, #b91c1c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(234, 88, 12, 0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280 !important;
        margin-bottom: 2rem;
        font-weight: 400;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar styling with light orange-cream background */
    .css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff7ed 0%, #ffedd5 100%) !important;
        border-right: 1px solid #fed7aa !important;
    }
    
    .css-1d391kg .stMarkdown, .css-1lcbmhc .stMarkdown {
        color: #1f2937 !important;
    }
    
    /* Button styling with orange theme */
    .stButton > button {
        background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(234, 88, 12, 0.3) !important;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.2) !important;
    }
    
    /* Form styling */
    .stForm {
        background: rgba(255, 255, 255, 0.7) !important;
        padding: 2rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(254, 215, 170, 0.5) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(234, 88, 12, 0.1) !important;
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #fed7aa !important;
        border-radius: 8px !important;
    }
    
    .stMultiSelect > div > div {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #fed7aa !important;
        border-radius: 8px !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border: 2px dashed #ea580c !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        text-align: center !important;
    }
    
    .stFileUploader label {
        color: #ea580c !important;
        font-weight: 600 !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #ea580c !important;
    }
    
    .stSlider > div > div > div {
        background-color: rgba(254, 215, 170, 0.5) !important;
    }
    
    /* Progress bar styling */
    .stProgress .stProgress-bar {
        background: linear-gradient(90deg, #ea580c 0%, #dc2626 100%) !important;
    }
    
    /* Detection stats box */
    .detection-stats {
        background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%) !important;
        padding: 2rem !important;
        border-radius: 16px !important;
        margin: 1.5rem 0 !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 8px 32px rgba(234, 88, 12, 0.2) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .detection-stats h3 {
        color: white !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        font-size: 1.5rem !important;
    }
    
    .detection-stats p {
        color: rgba(255, 255, 255, 0.95) !important;
        margin: 0.5rem 0 !important;
        font-weight: 500 !important;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid #10b981 !important;
        border-radius: 8px !important;
        color: #065f46 !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid #dc2626 !important;
        border-radius: 8px !important;
        color: #7f1d1d !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid #f59e0b !important;
        border-radius: 8px !important;
        color: #78350f !important;
    }
    
    /* Video player styling */
    video {
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(31, 41, 55, 0.1) !important;
        border: 1px solid rgba(254, 215, 170, 0.3) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border-radius: 8px !important;
        border: 1px solid #fed7aa !important;
        font-weight: 600 !important;
        color: #1f2937 !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.4) !important;
        border: 1px solid #fed7aa !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.6) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: 1px solid #fed7aa !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #ea580c !important;
    }
    
    /* Download button specific styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.2) !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.3) !important;
    }
    
    /* Header sections */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

def get_sample_videos():
    """Get sample video configurations"""
    return {
        "sv1.mp4": {
            "name": "Traffic Scene",
            "icon": "🚗",
            "objects": ["car", "truck", "bus"],
            "confidence": 0.5,
            "description": "Urban traffic with various vehicles including cars, trucks, and buses"
        },
        "sv2.mp4": {
            "name": "Horse & People",
            "icon": "🐎", 
            "objects": ["horse", "person"],
            "confidence": 0.6,
            "description": "Equestrian activity scene with horses and people"
        },
        "sv3.mp4": {
            "name": "Street Scene",
            "icon": "🚶",
            "objects": ["car", "person"],
            "confidence": 0.7,
            "description": "Street scene with cars and pedestrians walking"
        },
        "sv4.mp4": {
            "name": "Cycling Scene", 
            "icon": "🚴",
            "objects": ["bicycle", "person", "car"],
            "confidence": 0.7,
            "description": "Mixed traffic scene with cyclists, cars, and people"
        },
        "sv5.mp4": {
            "name": "City Traffic",
            "icon": "🌆",
            "objects": ["car", "bus", "bicycle", "person"],
            "confidence": 0.7,
            "description": "Busy city intersection with multiple object types"
        }
    }

def clean_output_files():
    """Clean up previous output files"""
    if os.path.exists("output.mp4"):
        os.remove("output.mp4")

def process_video(video_path, model, selected_objects, min_confidence, progress_bar=None):
    """Process video with object detection"""
    video_stream = cv2.VideoCapture(video_path)
    
    if not video_stream.isOpened():
        st.error("Error opening video file")
        return None, None
    
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Clean up previous output
    clean_output_files()
    
    # Use H.264 codec for better web compatibility
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    output_path = "output.mp4"
    
    # Try H.264 first, fall back to mp4v if not available
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # If H.264 failed, try mp4v
    if not out_video.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # If still failed, try XVID
    if not out_video.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = "output.avi"  # Change extension for XVID
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out_video.isOpened():
        st.error("Failed to initialize video writer")
        return None, None
    
    frame_count = 0
    detection_stats = {"total_detections": 0, "frames_processed": 0}
    
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break
        
        frame_count += 1
        detection_stats["frames_processed"] = frame_count
        
        # Update progress
        if progress_bar:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        frame_detections = 0
        if len(results[0].boxes) > 0:
            for detection in results[0].boxes.data:
                x0, y0 = (int(detection[0]), int(detection[1]))
                x1, y1 = (int(detection[2]), int(detection[3]))
                score = float(detection[4])
                cls = int(detection[5])
                object_name = model.names[cls]
                
                if object_name in selected_objects and score >= min_confidence:
                    frame_detections += 1
                    detection_stats["total_detections"] += 1
                    
                    # Draw bounding box with orange color
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (12, 88, 234), 2)  # Orange color in BGR
                    
                    # Draw label with orange background
                    label = f'{object_name}: {score:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x0, y0 - label_size[1] - 10), 
                                 (x0 + label_size[0], y0), (12, 88, 234), -1)  # Orange background
                    cv2.putText(frame, label, (x0, y0 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame info with better styling
        frame_info = f'Frame: {frame_count}/{total_frames} | Detections: {frame_detections}'
        cv2.putText(frame, frame_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (31, 41, 55), 2)  # Dark gray color
        
        out_video.write(frame)
    
    video_stream.release()
    out_video.release()
    
    return output_path, detection_stats

def render_sample_videos():
    """Render sample video selection in sidebar"""
    st.markdown("### 🎬 Sample Videos")
    
    sample_videos = get_sample_videos()
    
    # Create tabs for sample videos
    video_names = [f"{config['icon']} {config['name']}" for config in sample_videos.values()]
    
    selected_tab = st.radio(
        "Select Sample Video:",
        options=list(range(len(video_names))),
        format_func=lambda x: video_names[x],
        key="sample_video_selector"
    )
    
    # Get selected video info
    video_keys = list(sample_videos.keys())
    selected_video_key = video_keys[selected_tab]
    selected_video_config = sample_videos[selected_video_key]
    
    # Show description in expander
    with st.expander("📋 Video Description"):
        st.write(f"**Objects:** {', '.join(selected_video_config['objects'])}")
        st.write(f"**Confidence:** {int(selected_video_config['confidence']*100)}%")
        st.write(f"**Description:** {selected_video_config['description']}")
    
    # Use sample video button
    if st.button("🎬 Use This Sample", use_container_width=True):
        # Check for video file in multiple possible locations
        possible_paths = [
            f"sample-videos/{selected_video_key}",
            f".sample-videos/{selected_video_key}",
            f"videos/{selected_video_key}",
            selected_video_key
        ]
        
        video_found = False
        for path in possible_paths:
            if os.path.exists(path):
                st.session_state.selected_video_path = path
                st.session_state.selected_objects = selected_video_config['objects']
                st.session_state.min_confidence = selected_video_config['confidence']
                st.success(f"✅ Loaded {selected_video_config['name']}")
                video_found = True
                break
        
        if not video_found:
            st.error(f"❌ Sample video not found. Please check if the video file exists in any of these locations:")
            for path in possible_paths:
                st.write(f"- {path}")

def app():
    # Load custom CSS with creamy Claude.ai theme
    load_custom_css()
    
    # Main header
    st.markdown('<div class="main-header">🎯 Object Detection Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Powered by YOLOv8 | Advanced AI Vision Technology</div>', unsafe_allow_html=True)
    
    # Initialize YOLO model
    try:
        with st.spinner('🔄 Loading AI Model...'):
            model = YOLO('yolov8n.pt')
        object_names = list(model.names.values())
        st.success('✅ AI Model Loaded Successfully!')
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        render_sample_videos()
    
    # Main content
    st.markdown("### 📹 Video Processing")
    
    # Main form
    with st.form("detection_form"):
        # File upload
        uploaded_file = st.file_uploader(
            "📤 Upload Your Video",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        # Object selection
        default_objects = st.session_state.get('selected_objects', ['person', 'car'])
        selected_objects = st.multiselect(
            '🎯 Select Objects to Detect',
            object_names,
            default=default_objects,
            help="Choose which objects you want the AI to detect"
        )
        
        # Confidence slider
        default_confidence = st.session_state.get('min_confidence', 0.5)
        min_confidence = st.slider(
            '📊 Confidence Threshold',
            min_value=0.0,
            max_value=1.0,
            value=default_confidence,
            step=0.05,
            help="Higher values = more confident detections only"
        )
        
        # Submit button
        process_button = st.form_submit_button(
            "🚀 Start Detection",
            use_container_width=True
        )
    
    # Process video
    video_to_process = None
    
    if process_button:
        if uploaded_file is not None:
            # Save uploaded file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            video_to_process = temp_path
        elif 'selected_video_path' in st.session_state:
            video_to_process = st.session_state.selected_video_path
        else:
            st.warning("⚠️ Please upload a video or select a sample video from the sidebar")
    
    if video_to_process:
        st.markdown("---")
        st.markdown("### 🔄 Processing Video...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        try:
            with st.spinner('🤖 AI is analyzing your video...'):
                output_path, stats = process_video(
                    video_to_process, 
                    model, 
                    selected_objects, 
                    min_confidence,
                    progress_bar
                )
            
            if output_path and os.path.exists(output_path):
                st.success("🎉 Video processing completed!")
                
                # Show statistics
                st.markdown(f"""
                <div class="detection-stats">
                    <h3>📊 Detection Results</h3>
                    <p><strong>Total Detections:</strong> {stats['total_detections']}</p>
                    <p><strong>Frames Processed:</strong> {stats['frames_processed']}</p>
                    <p><strong>Objects Tracked:</strong> {', '.join(selected_objects)}</p>
                    <p><strong>Confidence Threshold:</strong> {int(min_confidence*100)}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display processed video
                st.markdown("### 🎬 Processed Video")
                st.video(output_path)
                
                # Download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=file.read(),
                        file_name="detected_objects.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            # Clean up temporary files
            if uploaded_file and os.path.exists(video_to_process):
                os.remove(video_to_process)
                
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            if uploaded_file and os.path.exists(video_to_process):
                os.remove(video_to_process)

if __name__ == "__main__":
    app()