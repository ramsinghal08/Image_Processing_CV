import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display

video_path = '/Users/ramsinghal/Downloads/MRT/Assignment 4 - Classical CV/Mallet_videos/IMG_9105.MOV'

def run_pipeline(video_path, hsv_values, kernel_size=5):
    cap = cv2.VideoCapture(video_path)

    # Setup the Display Widget
    video_widget = widgets.Image(format='jpeg')
    display(video_widget)

    lower = np.array([hsv_values[0], hsv_values[2], hsv_values[4]])
    upper = np.array([hsv_values[1], hsv_values[3], hsv_values[5]])
    morph_kernel = np.ones((kernel_size, kernel_size), np.uint8)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, morph_kernel)

        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

        colored_mask = mask_bgr.copy()
        colored_mask[np.where((colored_mask==[255,255,255]).all(axis=2))] = [0,255,0]
        result = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # Stack: [ Original | Binary Mask | Final Result ]
        final_display = np.hstack([frame, mask_bgr, result])

        # Update Widget
        _, encoded = cv2.imencode('.jpg', final_display)
        video_widget.value = encoded.tobytes()


tuned_values = [5, 25, 90, 255, 70, 255]
run_pipeline(video_path, tuned_values)
