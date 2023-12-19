import cv2
import os
import numpy as np
import math

def brenner(img):
    '''
    :param img:narray             the clearer the image,the larger the return value
    :return: float 
    '''
    shape = np.shape(img)
    
    out = 0
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-2):
            
            out+=(int(img[x+2,y])-int(img[x,y]))**2
            
    return out

def Laplacian(img):
    
    return cv2.Laplacian(img,cv2.CV_64F).var()

def SMD(img):
    
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[1]-1):
        for x in range(0, shape[0]-1):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

def SMD2(img):
    
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[1]-1):
        for x in range(0, shape[0]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

def variance(img):
    
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for y in range(0,shape[1]):
        for x in range(0,shape[0]):
            out+=(img[x,y]-u)**2
    return out

def energy(img):
 
    shape = np.shape(img)
    out = 0
    for y in range(0, shape[1]-1):
        for x in range(0, shape[0]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    return out

def Vollath(img):
  
    
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for y in range(0, shape[1]):
        for x in range(0, shape[0]-1):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def entropy(img):

    [rows, cols] = img.shape
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
    return out

def Tenengrad(image):
    '''
    Tenengrad gradient function, use Sobel operator to get the gradient of horizontal and vertical gradient value.
    :param image:
    :return:
    '''
    assert image is not None
    gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobel_x = cv2.Sobel(gray_image,cv2.CV_32FC1,1,0)
    sobel_y = cv2.Sobel(gray_image,cv2.CV_32FC1,0,1)
    sobel_xx = cv2.multiply(sobel_x,sobel_x)
    sobel_yy = cv2.multiply(sobel_y,sobel_y)
    image_gradient = sobel_xx + sobel_yy
    image_gradient = np.sqrt(image_gradient).mean()
    return image_gradient

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

            



