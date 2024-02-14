from optimization.Optimizer import Optimizer
from clarity.Clarity import InverseEntropyClarityMetric, EntropyClarityMetric
from microscope.Microscope import DummyCropMicroscopeController, DummyMultiFocusMicroscopeController
import cv2

def main2():
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


def main():
    controller = DummyMultiFocusMicroscopeController()
    
    for _ in range(5):
        img = controller.get_image().get_image_tensor()
        cv2.imshow(controller.get_current_focus(),img)
        cv2.waitKey(0)
    
    controller.move(2)
    for _ in range(5):
        img = controller.get_image().get_image_tensor()
        cv2.imshow(controller.get_current_focus(),img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()