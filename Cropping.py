from ultralytics import YOLO
import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from MicroscopeImage import MicroscopeImage

class Image_Cropper():
    def __init__(self, model_weights : str) -> None:
        """
        Class for cropping the images

        model_weights: path to the weights of the model
        """

        self.model = YOLO(model_weights)

        # Example usage:
    
    def crop(self, input_image, save_path, save=True):
        """
        Crop the image

        image_path: path to the input image
        save_path: path to save the cropped image
        """
        height = input_image.shape[0]
        width = input_image.shape[1]   

        predictions = self.model.predict(input_image)

        crop_path = save_path 
        crop_margin = 15    
        crops = []

        for i, box in enumerate(predictions[0].boxes.xyxy):
            # Get indices where the class index is equal to 0
            if predictions[0].boxes.cls[i] == 0:
                box_coordinates = box.cpu().numpy()
            else:
                 continue
            x1, y1, x2, y2 = map(int, box_coordinates)
            # Make the region 10 pixels wider
            x1 -= crop_margin
            y1 -= crop_margin
            x2 += crop_margin
            y2 += crop_margin

            # Ensure the coordinates are within the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # crop the image accrding bounding box
            rbg_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            cropped_image = rbg_image[y1:y2, x1:x2]
            # save crops in current image 
            if save:
    
                if not os.path.exists(crop_path):
                    os.makedirs(crop_path)
                crop_filename = f'{crop_path}/{i}.jpg'
                cv2.imwrite(crop_filename, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

            # save the list of crops for later evaluation 
            crops.append(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY))
        
        microscope_image_crops = [MicroscopeImage() for _ in range(len(crops))]
        for i, crop in enumerate(crops):
            microscope_image_crops[i].set_image_tensor(crop)

        return microscope_image_crops
