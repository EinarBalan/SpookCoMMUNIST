import cv2
import os

def overlay_images(background_folder, foreground_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    background_images = [f for f in os.listdir(background_folder) if os.path.isfile(os.path.join(background_folder, f))]
    foreground_images = [f for f in os.listdir(foreground_folder) if os.path.isfile(os.path.join(foreground_folder, f))]

    for bg_image_name in background_images:
        bg_image_path = os.path.join(background_folder, bg_image_name)
        original_bg_image = cv2.imread(bg_image_path)
        if original_bg_image is None: continue

        for fg_image_name in foreground_images:
            fg_image_path = os.path.join(foreground_folder, fg_image_name)
            fg_image = cv2.imread(fg_image_path, cv2.IMREAD_UNCHANGED)
            
            if fg_image is None: continue

            if fg_image.shape[2] == 4:  # Check if the foreground image has an alpha channel
                # Create a copy of the background image to avoid continuous overlays
                bg_image = original_bg_image.copy()

                # Resize the background image to match the foreground image dimensions
                bg_image = cv2.resize(bg_image, (fg_image.shape[1], fg_image.shape[0]))

                alpha_channel = fg_image[:, :, 3] / 255.0
                for c in range(0, 3):
                    bg_image[:, :, c] = alpha_channel * fg_image[:, :, c] + (1 - alpha_channel) * bg_image[:, :, c]

                output_image_path = os.path.join(output_folder, f"{os.path.splitext(bg_image_name)[0]}_{fg_image_name}")
                cv2.imwrite(output_image_path, bg_image)
                
background_folder = './raw/sepia'
foreground_folder = './foreground/communists'
output_folder = './spookcommunist/all/communist/sepia'
overlay_images(background_folder, foreground_folder, output_folder)

background_folder = './raw/sepia'
foreground_folder = './foreground/spooks'
output_folder = './spookcommunist/all/spook/sepia'
overlay_images(background_folder, foreground_folder, output_folder)

background_folder = './raw/woods'
foreground_folder = './foreground/communists'
output_folder = './spookcommunist/all/communist/woods'
overlay_images(background_folder, foreground_folder, output_folder)

background_folder = './raw/woods'
foreground_folder = './foreground/spooks'
output_folder = './spookcommunist/all/spook/woods'
overlay_images(background_folder, foreground_folder, output_folder)