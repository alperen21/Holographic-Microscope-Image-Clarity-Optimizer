from Optimizer import Optimizer
from Clarity import LaplacianClarityMetric
from Microscope import RestMicroscopeController
import cv2

def main():
    laplacian = LaplacianClarityMetric("yolov8n.pt")
    optimizer = Optimizer(
        clarity_metric= laplacian,
        microscope_controller=RestMicroscopeController(),
        lr=1
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", laplacian(optimized_image))

if __name__ == "__main__":
    main()