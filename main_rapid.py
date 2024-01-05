from optimization.Optimizer import RapidOptimizer
from clarity.Clarity import InferenceClarityMetric
from microscope.Microscope import DummyClinicalMicroscopeController
import cv2

def main():
    optimizer = RapidOptimizer(
        microscope_controller=DummyClinicalMicroscopeController()
    )
    optimized_image = optimizer.start()
    cv2.imshow("",optimized_image.get_image_tensor())
    cv2.waitKey(0)
    



if __name__ == "__main__":
    main()