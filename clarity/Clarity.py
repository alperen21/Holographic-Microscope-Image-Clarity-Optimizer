from typing import Any
import cv2
import os
import numpy as np
from scipy.stats import hmean,gmean
import math
from multiprocessing import Pool
from abc import ABC, abstractclassmethod
from cropping.Cropping import Image_Cropper
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import Compose, Resize, Normalize
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
from tensorflow.keras.models import load_model, Model




clarity_model = models.resnet18(pretrained=False)
# Modify the last fully connected layer for your specific task
num_ftrs = clarity_model.fc.in_features
clarity_model.fc = nn.Linear(num_ftrs, 1)  # Assuming the output feature size is 1

clarity_model.load_state_dict(torch.load('clarity_model.pt'))


class LinearClarityMetric(ABC):
    def __init__(self, model_weights, crop=False) -> None:
        self.image_cropper = Image_Cropper(model_weights)
        self.crop_images = crop

    def crop(self, image):
        if self.crop_images:
            return self.image_cropper.crop(image, "results", save=False)
        else:
            return [image]
    
    @abstractclassmethod
    def calculate_clarity(self, image):
        pass 
    
    @abstractclassmethod
    def get_max_value(self):
        pass

    def get_clarity(self, image):
        crops = self.crop(image)

        if len(crops) == 0:
            return 0
        
        clarity_score = 0
        for crop in crops:
            clarity_score += self.calculate_clarity(crop)
        
        average_clarity = clarity_score/len(crops)
        return average_clarity

    def get_normalized_clarity(self, image):
        crops = self.crop(image)
        
        clarity_score = 0

        if len(crops) == 0:
            return 0
        for crop in crops:
            score = self.calculate_clarity(crop) / self.get_max_value()
            clarity_score += score
        
        average_clarity = clarity_score/len(crops)
        return average_clarity
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.get_clarity(*args, **kwds)


class KerasClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)
        self.model = load_model("focused_unfocused_model.h5")
    
    def calculate_clarity(self, image):
        image = image.get_image_tensor()
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0


        layer_name = self.model.layers[-1].name  # Name of the last layer
        intermediate_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        activations = intermediate_model.predict(image)

        first_values = activations[:, 0] 

        return first_values[0]

    
    def get_max_value(self):
        return 1


class ConvolutionalClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)
        self.transform = Compose([
            Resize((224, 224), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calculate_clarity(self, image):
        global clarity_model
        image = image.get_image_tensor()
        if len(image.shape) == 2:
            # Add a channel dimension (HxW -> HxWx1)
            image = np.expand_dims(image, axis=2)
            image = np.repeat(image, 3, axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1)  
        image = convert_image_dtype(image, dtype=torch.float32)

        # image = convert_image_dtype(image, dtype=torch.float)
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add a batch dimension

        return clarity_model(image)

    def get_max_value(self):
        return 10

class LaplacianClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return Laplacian(image)

    def get_max_value(self):
        return 255**2
    
class BrennerClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return brenner(image)

    def get_max_value(self):
        return 255**2

class SMDClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return SMD(image)

    def get_max_value(self):
        return 255

class SMD2ClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return SMD2(image)

    def get_max_value(self):
        return 255
    
class VarianceClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return variance(image)

    def get_max_value(self):
        return 255**2

class EnergyClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return energy(image)

    def get_max_value(self):
        return 255**2

class VollathClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return Vollath(image)

    def get_max_value(self):
        return 255**2
    

class InverseEntropyClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return -entropy(image)

    def get_max_value(self):
        return np.log2(256)

class EntropyClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return entropy(image)

    def get_max_value(self):
        return np.log2(256)

class TenengradClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)

    def calculate_clarity(self, image):
        return Tenengrad(image)

    def get_max_value(self):
        return 255**2

class HarmonicMeanClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)
    
    def calculate_clarity(self, image):
        return harmonic_mean(image)

    def get_max_value(self):
        return 1

class GeometricMeanClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)
    
    def calculate_clarity(self, image):
        return geometric_mean(image)

    def get_max_value(self):
        return 1

class ArithmeticMeanClarityMetric(LinearClarityMetric):
    def __init__(self, model_weights) -> None:
        super().__init__(model_weights)
    
    def calculate_clarity(self, image):
        return arithmetic_mean(image)

    def get_max_value(self):
        return 1



def brenner(img):
    '''
    :param img:narray             the clearer the image,the larger the return value
    :return: float 
    '''
    img = img.get_image_tensor()
    shape = np.shape(img)
    
    out = 0
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-2):
            for c in range(shape[2]):  # Loop over each channel
                out+=(int(img[x+2,y,c])-int(img[x,y,c]))**2
            
    max_brenner_value = 255**2 * img.size  # This is an approximation
    normalized_brenner_value = out / max_brenner_value

    return normalized_brenner_value
def Laplacian(img):
    # Calculate the Laplacian value
    img = img.get_image_tensor()
    laplacian_value = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Normalize the Laplacian value
    max_laplacian_value = 255**2  # This is an approximation
    normalized_laplacian_value = laplacian_value / max_laplacian_value

    return normalized_laplacian_value

def calculate_metric(metric, img):
    return metric(img)

def harmonic_mean(img):
    img = img.get_image_tensor()
    metrics = [brenner, Laplacian, SMD, SMD2, variance, energy, Vollath, Tenengrad, entropy]
    pool = Pool()
    results = [pool.apply(calculate_metric, args=(metric, img)) for metric in metrics]
    pool.close()
    pool.join()

    return hmean(results)

def geometric_mean(img):
    img = img.get_image_tensor()
    metrics = [brenner, Laplacian, SMD, SMD2, variance, energy, Vollath, Tenengrad, entropy]
    pool = Pool()
    results = [pool.apply(calculate_metric, args=(metric, img)) for metric in metrics]
    pool.close()
    pool.join()

    return gmean(results)

def arithmetic_mean(img):
    img = img.get_image_tensor()
    metrics = [brenner, Laplacian, SMD, SMD2, variance, energy, Vollath, Tenengrad, entropy]
    pool = Pool()
    results = [pool.apply(calculate_metric, args=(metric, img)) for metric in metrics]
    pool.close()
    pool.join()

    return np.mean(results)

def SMD(img):
    
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        out+=math.fabs(int(img[x])-int(img[x-1]))

    # Normalize the SMD value
    max_smd_value = 255 * img.size  # This is an approximation
    normalized_smd_value = out / max_smd_value

    return normalized_smd_value

def SMD2(img):
    
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[0]-1):
        for x in range(0, shape[0]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    
    max_smd_value = 255 * img.size  # This is an approximation
    normalized_smd_value = out / max_smd_value

    return normalized_smd_value

def variance(img):
    img = img.get_image_tensor()
    u = np.mean(img)
    variance_value = np.var(img)
    max_variance_value = 255**2  # This is an approximation
    normalized_variance_value = variance_value / max_variance_value
    return normalized_variance_value

def energy(img):
 
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        out+=((int(img[x+1])-int(img[x]))**2)*((int(img[x+2])-int(img[x]))**2)
    return out

def Vollath(img):
    
    img = img.get_image_tensor()
    
    # Convert img to grayscale if it is a color image
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-1):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def entropy(img):

    img = img.get_image_tensor()

    if len(img.shape) == 3:
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape

    # [rows, cols] = img.shape
    h = 0
    hist_gray = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    # hn valueis not correct
    hb = np.zeros((256, 1), np.float32)
    #hn = np.zeros((256, 1), np.float32)
    for j in range(0, 256):
        hb[j, 0] = hist_gray[j, 0] / (rows*cols)
    for i in range(0, 256):
        if hb[i, 0] > 0:
            h = h - (hb[i, 0])*math.log(hb[i, 0],2)
                
    out = h
    max_entropy_value = np.log2(256)
    return out / max_entropy_value

def Tenengrad(image):
    '''
    Tenengrad gradient function, use Sobel operator to get the gradient of horizontal and vertical gradient value.
    :param image:
    :return:
    '''
    image = image.get_image_tensor()
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobel_x = cv2.Sobel(gray_image,cv2.CV_32FC1,1,0)
    sobel_y = cv2.Sobel(gray_image,cv2.CV_32FC1,0,1)
    sobel_xx = cv2.multiply(sobel_x,sobel_x)
    sobel_yy = cv2.multiply(sobel_y,sobel_y)
    image_gradient = sobel_xx + sobel_yy
    image_gradient = np.sqrt(image_gradient).mean()

    max_tenengrad_value = 255**2  # This is an approximation
    return image_gradient / max_tenengrad_value

def LinearClarityMetric(crops, metric_name):
    
    sharpness_sum = 0
    for img in crops:
        if metric_name == 'Brenner':
            clarity_score = brenner(img)
        elif metric_name == 'Laplacian':
            clarity_score = Laplacian(img)
        elif metric_name == 'SMD':
            clarity_score = SMD(img)
        elif metric_name == 'SMD2':
            clarity_score = SMD2(img)
        elif metric_name == 'Variance':
            clarity_score = variance(img)
        elif metric_name == 'Energy':
            clarity_score = energy(img)
        elif metric_name == 'Vollath':
            clarity_score = Vollath(img)
        elif metric_name == 'Entropy':
            clarity_score = entropy(img)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        clarity_score += clarity_score
        average_clarity = sharpness_sum/len(crops)
        return average_clarity


def DummyClarityMetric(img):
    img_focus = img.get_focus()
    focus_to_clarity = {
        -3.0 : 10,
        -2.5 : 10.5,
        -2.0 : 11,
        -1.5 : 11.5,
        -1.0 : 12,
        -0.5 : 12.5,
        0 : 13,
        0.5 : 12.5,
        1.0 : 12,
        1.5 : 11.5,
        2.0 : 11,
        2.5 : 10.5,
        3.0 : 10
    }
    return focus_to_clarity[img_focus]

if __name__ == "__main__":
    # c = DummyClarityMetric
    c = LinearClarityMetric

            



