import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

#replace folder path here
folder_path = '/Users/ramsinghal/Downloads/MRT/Assignment 4 - Classical CV/Cone_images'
image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) +
                     glob.glob(os.path.join(folder_path, '*.png')))

#change img_id here for diff. image to process from the folder path above
img_id = 0

def group_and_hull_contours(contours, distance_threshold=50):
    """
    Groups contours that are close to each other and wraps them in a single Convex Hull.
    This solves the 'split cone' problem regardless of angle or alignment.
    """
    if not contours: return []

    # Calculate centroids for all contours
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        centroids.append((cx, cy))

    # Simple clustering: If centroids are close, group them
    # A list of lists, where each inner list is a group of contour indices
    groups = []
    processed = [False] * len(contours)

    for i in range(len(contours)):
        if processed[i]: continue

        current_group = [i]
        processed[i] = True

        # Check all other contours
        for j in range(i + 1, len(contours)):
            if processed[j]: continue

            # Calculate Euclidean distance between centroids
            dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + (centroids[i][1] - centroids[j][1])**2)

            if dist < distance_threshold:
                current_group.append(j)
                processed[j] = True

        groups.append(current_group)

    # Now create a Convex Hull for each group
    final_boxes = []
    for group in groups:
        # Combine all points from all contours in this group
        combined_points = np.vstack([contours[idx] for idx in group])

        # Calculate Convex Hull of the group
        hull = cv2.convexHull(combined_points)

        # Get bounding box of the Hull
        rect = cv2.boundingRect(hull)
        final_boxes.append(rect)

    return final_boxes


def detect_cones(img_idx,a_min, a_max, b_min, b_max):
    if not image_files: return
    img = cv2.imread(image_files[img_idx])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    img_lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    lower = np.array([l_min, a_min, b_min])
    upper = np.array([l_max, a_max, b_max])
    mask = cv2.inRange(img_lab, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]

    final_boxes = group_and_hull_contours(valid_contours, distance_threshold=100)

    final_output = img_rgb.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    count = 0
    for box in final_boxes:
        x, y, w, h = box
        if w*h > 200:
            count += 1
            cv2.rectangle(final_output, (x, y), (x+w, y+h), (0, 255, 0), 2)


    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title("LAB Mask (Fragmented)")
    plt.axis('off')


    plt.subplot(1, 2, 2)
    plt.imshow(final_output)
    plt.title(f"Results: {count} Cones")
    plt.axis('off')

    plt.show()

detect_cones(img_id, 135, 255, 0, 255)
