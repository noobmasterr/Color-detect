
import cv2
import numpy as np

def preprocess_mask(mask):
    kernel_opening = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
    kernel_closing = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closing)
    return mask

def track_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'green': ([30, 50, 50], [90, 255, 255]),
        'blue': ([100, 100, 100], [140, 255, 255]),
        'yellow': ([20, 100, 100], [40, 255, 255]),
        'pink': ([140, 100, 100], [170, 255, 255])
    }

    masks = {}

    for color_name, (lower_color, upper_color) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))
        mask = preprocess_mask(mask)
        masks[color_name] = mask

    result_frame = np.zeros_like(frame)

    contour_colors = {
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'pink': (255, 0, 255)
    }

    for color_name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_frame, contours, -1, contour_colors[color_name], 2)
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(result_frame, (cx, cy), 5, contour_colors[color_name], -1)

    cv2.imshow('Result', result_frame)
    return masks

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = track_color(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
