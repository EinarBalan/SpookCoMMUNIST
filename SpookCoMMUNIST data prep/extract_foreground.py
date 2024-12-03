import os
import cv2
from rembg import remove
from PIL import Image

def extract_foreground(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            # Read the image
            image = Image.open(input_path)

            # Remove the background
            output_image = remove(image)

            # Save the output image
            output_image.save(output_path)

if __name__ == "__main__":
    input_folder = './raw/communists'
    output_folder = './foreground/communists'
    extract_foreground(input_folder, output_folder)
    
    input_folder = './raw/spooks'
    output_folder = './foreground/spooks'
    extract_foreground(input_folder, output_folder)