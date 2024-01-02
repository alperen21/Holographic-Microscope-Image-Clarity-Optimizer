from optimization.Optimizer import Optimizer
from clarity.Clarity import InverseEntropyClarityMetric, EntropyClarityMetric
from microscope.Microscope import DummyCropMicroscopeController
import cv2

def main():
    clarityMetric = EntropyClarityMetric("24_10_23_yolov8x_no_aug_iou_0.7.pt")
    optimizer = Optimizer(
        clarity_metric= clarityMetric,
        microscope_controller=DummyCropMicroscopeController(),
        lr=10
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", clarityMetric(optimized_image))
    cv2.imshow("",optimized_image.get_imageclarityMetric_tensor())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()