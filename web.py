import cv2
import numpy as np
import imutils
import time
import random
import pandas as pd
import requests

# Load the Haar cascade for object detection
cascade = cv2.CascadeClassifier('path_to_haar_cascade.xml')  # Provide the correct path

# Function to apply random defects to an image
def apply_random_defects(image):
    # Simulate defects by adding noise, lines, or other modifications
    # You can customize this function to introduce various types of defects
    
    # Example: Adding random noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    
    return noisy_image

def main():
    # Initialize webcam capture
    url = "http://100.77.229.254:8080/shot.jpg"
    defects = ["Scratch", "Crack", "Dent", "Stain", "Discoloration", "Chip", "Warp", "Bulge", "Rough Surface", "Pit"]
    color_change_interval = 7  # Change color every 7 seconds
    last_color_change = time.time()
    current_color = (0, 255, 0)  # Initial color

    defects_record = []  # List to record defects
    random_defect = ""
    
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        
        
        # Apply random defects to the captured frame
        frame_with_defects = apply_random_defects(frame)
        
        # Convert the frame to grayscale for object detection
        gray = cv2.cvtColor(frame_with_defects, cv2.COLOR_BGR2GRAY)
        
        # Perform object detection using the Haar cascade
        objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the detected objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame_with_defects, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the frame with defects and detected objects
        cv2.imshow('Defected Product', frame_with_defects)
        
        # Exit the loop on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
   
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
