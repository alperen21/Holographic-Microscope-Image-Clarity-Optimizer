from Optimizer import Optimizer
from Clarity import DummyClarityMetric, LaplacianClarityMetric
from Microscope import DummyMicroscopeController, DummyCropMicroscopeController
import cv2

def main():
    laplacian = LaplacianClarityMetric("yolov8n.pt")
    optimizer = Optimizer(
        clarity_metric= laplacian,
        microscope_controller=DummyCropMicroscopeController(),
        lr=1
    )
    optimized_image = optimizer.start()
    cv2.imshow("optimized image", optimized_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()