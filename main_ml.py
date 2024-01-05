from optimization.Optimizer import GradOptimizer
from clarity.Clarity import InferenceClarityMetric
from microscope.Microscope import DummyClinicalMicroscopeController
import cv2

def main():
    clarityMetric = InferenceClarityMetric("24_10_23_yolov8x_no_aug_iou_0.7.pt")
    optimizer = GradOptimizer(
        clarity_metric= clarityMetric,
        microscope_controller=DummyClinicalMicroscopeController(),
        lr=1
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", clarityMetric(optimized_image))
    cv2.imshow("",optimized_image.get_image_tensor())
    cv2.waitKey(0)
    



if __name__ == "__main__":
    main()