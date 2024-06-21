# Clone your forked repository
!git clone https://github.com/YOUR_GITHUB_USERNAME/Object-tracking-and-counting-using-YOLOV8.git

# Navigate into the repository
%cd /content/Object-tracking-and-counting-using-YOLOV8

# Import necessary libraries
import os
from datetime import datetime
from your_script import YOLOv8_ObjectCounter  # Replace with the actual script name

# Main Execution
source_ref = 'B1'  # Code Panneau
current_date = datetime.now().strftime('%Y-%m-%d')  # Get current date
counter = YOLOv8_ObjectCounter('yolov8s.pt', conf=0.60, iou=0.60)
counter.predict_video(
    video_source=0,  # Use 0 for the default camera
    output_file_path=f'/content/{source_ref}-{current_date}.csv',
    frame_skip=2,
    update_interval=5  # Update CSV every 5 seconds
)
