from Cropping import Image_Cropper
import os
import cv2
from Clarity import Laplacian

def main():
    image_cropper = Image_Cropper("yolov5n.pt")
    images = image_cropper.crop(cv2.imread(os.path.join("crop_images", "input.jpg")), "results", save=False)

    print(Laplacian(images[0]))



    

if __name__ == "__main__":
    main()
