from Optimizer import Optimizer
from Clarity import DummyClarityMetric, Laplacian
from Microscope import DummyMicroscopeController
import cv2

def main():
    optimizer = Optimizer(
        clarity_metric=Laplacian,
        microscope_controller=DummyMicroscopeController(),
        lr=0.1
    )
    optimized_image = optimizer.start()
    cv2.imshow("optimized image", optimized_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()