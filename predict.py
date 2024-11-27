from ultralytics import YOLO
from PIL import Image
import os

# Initialize the YOLO model
modelx = YOLO("/home/athenaai/Documents/Daniyal/visa-german/captcha.v1i.yolov8/runs/detect/train3/weights/best.pt")

source = "/home/athenaai/Documents/Daniyal/visa-german/captcha.v1i.yolov8/images"
output_dir = "/home/athenaai/Documents/Daniyal/visa-german/captcha.v1i.yolov8/runs/predict"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Run inference on the source
results = modelx(source, stream=True)  # generator of Results objects
index = 0

for result in results:
    index += 1
    # Plot the result; returns an image in BGR format
    img_bgr = result.plot(conf=False)
    
    # Convert BGR to RGB
    im_rgb = Image.fromarray(img_bgr[..., ::-1])
    
    # Define the output file path
    output_path = os.path.join(output_dir, f"{index}.jpg")
    
    # Save the image
    im_rgb.save(output_path)  # Pass the path as a positional argument

    print(f"Saved {output_path}")
