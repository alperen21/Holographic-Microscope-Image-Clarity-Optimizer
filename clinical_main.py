from optimization.Optimizer import Optimizer
from clarity.Clarity import LaplacianClarityMetric, ConvolutionalClarityMetric, InverseEntropyClarityMetric, EntropyClarityMetric
from microscope.Microscope import DummyClinicalMicroscopeController
import cv2

def main():
    laplacian = InverseEntropyClarityMetric("24_10_23_yolov8x_no_aug_iou_0.7.pt")
    optimizer = Optimizer(
        clarity_metric= laplacian,
        microscope_controller=DummyClinicalMicroscopeController(),
        lr=20
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", laplacian(optimized_image))
    cv2.imshow("",optimized_image.get_image_tensor())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
