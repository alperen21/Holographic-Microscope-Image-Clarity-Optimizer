from optimization.Optimizer import Optimizer
from clarity.Clarity import LaplacianClarityMetric, ConvolutionalClarityMetric
from microscope.Microscope import DummyCropMicroscopeController
import cv2

def main():
    laplacian = LaplacianClarityMetric("yolov8n.pt")
    optimizer = Optimizer(
        clarity_metric= laplacian,
        microscope_controller=DummyCropMicroscopeController(),
        lr=1
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", laplacian(optimized_image))
    cv2.imshow("",optimized_image.get_image_tensor())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()