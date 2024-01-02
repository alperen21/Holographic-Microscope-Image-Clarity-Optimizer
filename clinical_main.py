from optimization.Optimizer import Optimizer
from clarity.Clarity import LaplacianClarityMetric, ConvolutionalClarityMetric, InverseEntropyClarityMetric, EntropyClarityMetric
from microscope.Microscope import DummyClinicalMicroscopeController
import cv2
from configurations.config import config

def main():
    clarityMetric = InverseEntropyClarityMetric()
    optimizer = Optimizer(
        clarity_metric= clarityMetric,
        microscope_controller=DummyClinicalMicroscopeController()
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", clarityMetric(optimized_image))
    cv2.imshow("",optimized_image.get_image_tensor())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
