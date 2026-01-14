import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

#replace with folder location
#folder_path = '/Users/ramsinghal/Downloads/MRT/Assignment 4 - Classical CV/Cone_images'
#replace with image index in folder
#img_index = 1

image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) +
                     glob.glob(os.path.join(folder_path, '*.png')))

def pre_processing(index, gamma):
    img = cv2.imread(image_files[index])

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(img, table)

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 2, 1); plt.imshow(enhanced_rgb); plt.title("Enhanced Image")
    plt.axis('off')
    plt.show()
    return enhanced_lab, enhanced_rgb

def process_edges(edges, img_rgb):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=2)
    edges_processed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    lines = cv2.HoughLinesP(edges_processed, 1, np.pi/180, threshold=25,
                            minLineLength=25, maxLineGap=40)

    line_img = img_rgb.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > 20:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.subplot(1, 1, 1); plt.imshow(line_img); plt.title("Detected Lines")
    plt.axis('off')
    plt.show()
    return edges_processed, line_img

def geometric_verification(contours):

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        aspect_ratio = float(h) / w
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        if solidity > 0.3 and aspect_ratio > 0.2:
            valid_contours.append(cnt)
            
    return valid_contours

edges, enhanced_rgb = detect_edge(img_index,1.5)
edges_processed, line_img = process_edges(edges, enhanced_rgb)

contours, _ = cv2.findContours(edges_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_contours = geometric_verification(contours)

cv2.drawContours(enhanced_rgb, valid_contours, -1, (0, 255, 0), 2)

for cnt in valid_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(enhanced_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Displays final output with contours and rectangles
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(line_img)
axes[0].set_title('Detected Lines (Hough)')
axes[0].axis('off')

axes[1].imshow(enhanced_rgb)
axes[1].set_title(f'Final Results ({len(valid_contours)} valid)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
