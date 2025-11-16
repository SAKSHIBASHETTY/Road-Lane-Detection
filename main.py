import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

class LaneDetector:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        
    def region_of_interest(self, img):
        """Define and apply a region of interest mask"""
        height, width = img.shape[:2]
        
        # Define polygon vertices
        vertices = np.array([[
            (width * 0.1, height),
            (width * 0.45, height * 0.6),
            (width * 0.55, height * 0.6),
            (width * 0.9, height)
        ]], dtype=np.int32)
        
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        
        return cv2.bitwise_and(img, mask)
    
    def detect_edges(self, frame):
        """Convert to grayscale and detect edges"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        return edges
    
    def detect_lane_lines(self, edges):
        """Use Hough Transform to detect lane lines"""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=150
        )
        
        return lines
    
    def separate_lines(self, lines, shape):
        """Separate lines into left and right lanes"""
        height, width = shape[:2]
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 - x1 == 0:
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope and position
            if slope < -0.5 and x1 < width / 2 and x2 < width / 2:
                left_lines.append(line[0])
            elif slope > 0.5 and x1 > width / 2 and x2 > width / 2:
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def fit_polynomial(self, lines, shape):
        """Fit a polynomial to the detected lines"""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        return np.polyfit(y_coords, x_coords, 2)
    
    def calculate_curvature(self, fit, y_eval, ym_per_pix=30/720, xm_per_pix=3.7/700):
        """Calculate the radius of curvature"""
        if fit is None:
            return 0
        
        # Calculate radius of curvature in meters
        curvature = ((1 + (2*fit[0]*y_eval*ym_per_pix + fit[1])**2)**1.5) / np.abs(2*fit[0])
        
        return curvature * xm_per_pix / (ym_per_pix ** 2)
    
    def calculate_offset(self, left_fit, right_fit, shape, xm_per_pix=3.7/700):
        """Calculate vehicle offset from lane center"""
        height, width = shape[:2]
        
        if left_fit is None or right_fit is None:
            return 0
        
        # Calculate lane positions at bottom of image
        left_x = left_fit[0] * height**2 + left_fit[1] * height + left_fit[2]
        right_x = right_fit[0] * height**2 + right_fit[1] * height + right_fit[2]
        
        # Calculate lane center
        lane_center = (left_x + right_x) / 2
        
        # Calculate vehicle center (assume camera is centered)
        vehicle_center = width / 2
        
        # Calculate offset in meters
        offset = (vehicle_center - lane_center) * xm_per_pix
        
        return offset
    
    def draw_lanes(self, frame, left_fit, right_fit):
        """Draw detected lanes on the frame"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        if left_fit is not None and right_fit is not None:
            # Generate y coordinates
            y_coords = np.linspace(0, height - 1, height)
            
            # Calculate x coordinates for left and right lanes
            left_x = left_fit[0] * y_coords**2 + left_fit[1] * y_coords + left_fit[2]
            right_x = right_fit[0] * y_coords**2 + right_fit[1] * y_coords + right_fit[2]
            
            # Create points for polygon
            left_points = np.array([np.column_stack([left_x, y_coords])], dtype=np.int32)
            right_points = np.array([np.flipud(np.column_stack([right_x, y_coords]))], dtype=np.int32)
            
            # Draw filled polygon between lanes
            points = np.concatenate([left_points[0], right_points[0]])
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            
            # Draw lane lines
            cv2.polylines(frame, left_points, False, (255, 0, 0), 3)
            cv2.polylines(frame, np.array([np.column_stack([right_x, y_coords])], dtype=np.int32), False, (0, 0, 255), 3)
        
        # Blend overlay with frame
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return result
    
    def process_frame(self, frame):
        """Process a single frame for lane detection"""
        height = frame.shape[0]
        
        # Detect edges
        edges = self.detect_edges(frame)
        
        # Apply region of interest
        roi_edges = self.region_of_interest(edges)
        
        # Detect lines
        lines = self.detect_lane_lines(roi_edges)
        
        # Separate left and right lines
        left_lines, right_lines = self.separate_lines(lines, frame.shape)
        
        # Fit polynomials
        left_fit = self.fit_polynomial(left_lines, frame.shape)
        right_fit = self.fit_polynomial(right_lines, frame.shape)
        
        # Use smoothing with previous fits
        if left_fit is not None:
            if self.left_fit is not None:
                left_fit = 0.7 * self.left_fit + 0.3 * left_fit
            self.left_fit = left_fit
        
        if right_fit is not None:
            if self.right_fit is not None:
                right_fit = 0.7 * self.right_fit + 0.3 * right_fit
            self.right_fit = right_fit
        
        # Draw lanes
        result = self.draw_lanes(frame, self.left_fit, self.right_fit)
        
        # Calculate curvature and offset
        curvature_left = self.calculate_curvature(self.left_fit, height)
        curvature_right = self.calculate_curvature(self.right_fit, height)
        curvature = (curvature_left + curvature_right) / 2
        
        offset = self.calculate_offset(self.left_fit, self.right_fit, frame.shape)
        
        # Add text overlay
        cv2.putText(result, f'Curvature: {curvature:.2f}m', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, f'Offset: {offset:.2f}m {"left" if offset > 0 else "right"}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result, curvature, offset

def process_video(video_path, progress_bar, status_text):
    """Process video file"""
    cap = cv2.VideoCapture(video_path)
    detector = LaneDetector()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, curvature, offset = detector.process_frame(frame)
        out.write(processed_frame)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f'Processing frame {frame_count}/{total_frames}')
    
    cap.release()
    out.release()
    
    return output_path

def main():
    st.set_page_config(page_title="Lane Detection System", layout="wide")
    
    st.title("🚗 Real-Time Lane Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Settings")
    input_mode = st.sidebar.radio("Select Input Mode", ["Upload Video", "Webcam"])
    
    if input_mode == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()  # Close the file before using it
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                st.video(tfile.name)
            
            with col2:
                st.subheader("Processed Video")
                
                if st.button("Process Video"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Processing video..."):
                        output_path = process_video(tfile.name, progress_bar, status_text)
                    
                    st.success("Video processed successfully!")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download Processed Video",
                            data=f,
                            file_name="lane_detected_output.mp4",
                            mime="video/mp4"
                        )
                    
                    # Store output path in session state for cleanup
                    if 'output_path' in st.session_state:
                        try:
                            os.unlink(st.session_state.output_path)
                        except:
                            pass
                    st.session_state.output_path = output_path
    
    else:  # Webcam mode
        st.subheader("Webcam Lane Detection")
        st.info("Click 'Start Webcam' to begin real-time lane detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Feed")
            original_frame = st.empty()
        
        with col2:
            st.subheader("Processed Feed")
            processed_frame = st.empty()
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            curvature_metric = st.empty()
        with metrics_col2:
            offset_metric = st.empty()
        
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        
        if start_button:
            cap = cv2.VideoCapture(0)
            detector = LaneDetector()
            
            st.session_state.webcam_running = True
            
            while st.session_state.get('webcam_running', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Process frame
                processed, curvature, offset = detector.process_frame(frame)
                
                # Display frames
                original_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                processed_frame.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Display metrics
                curvature_metric.metric("Lane Curvature", f"{curvature:.2f}m")
                offset_metric.metric("Vehicle Offset", f"{abs(offset):.2f}m {'left' if offset > 0 else 'right'}")
                
                if stop_button:
                    st.session_state.webcam_running = False
                    break
            
            cap.release()
    
    # Information section
    with st.sidebar:
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This application performs real-time lane detection using:
        - **Edge Detection**: Canny edge detection
        - **Line Detection**: Hough Transform
        - **Polynomial Fitting**: 2nd order polynomial
        - **Curvature Calculation**: Radius of curvature
        - **Offset Calculation**: Vehicle position relative to lane center
        """)
        
        st.markdown("---")
        st.subheader("Features")
        st.markdown("""
        ✅ Video upload processing  
        ✅ Real-time webcam detection  
        ✅ Lane curvature calculation  
        ✅ Vehicle offset measurement  
        ✅ Download processed video  
        """)

if __name__ == "__main__":
    main()