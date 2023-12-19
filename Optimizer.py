from Microscope import DummyMicroscopeController
from Clarity import DummyClarityMetric
import cv2

class Optimizer:
    def __init__(self, clarity_metric, microscope_controller, lr=0.01):
        """
        clarity_metric: function that takes in an image and returns a clarity score
        microscope_controller: object that controls the microscope
        lr: learning rate
        """
        self.lr = lr
        self.clarity_metric = clarity_metric
        self.microscope_controller = microscope_controller

    def forward(self):
        """
        Returns the clarity of the current image
        """
        image = self.microscope_controller.get_image()
        clarity = self.clarity_metric(image)
        return clarity

        
    def gradient(self, previous_clarity, move_amount=0.5):
        """
        Returns the gradient of the clarity metric with respect to the move amount
        """
        self.microscope_controller.move(move_amount)
        image = self.microscope_controller.get_image()
        new_clarity = self.clarity_metric(image)
        print(new_clarity)
        grad = new_clarity - previous_clarity
        self.microscope_controller.move((-1)*move_amount)

        
        if new_clarity < previous_clarity:
            self.microscope_controller.move((-1)*move_amount)
            image = self.microscope_controller.get_image()
            new_clarity_2 = self.clarity_metric(image)
            self.microscope_controller.move(move_amount)


            if new_clarity_2 < previous_clarity:
                return 0
        
        return grad
    
    def convergence_check(self, previous_clarity, current_clarity):
        """
        Returns True if the optimizer has converged, False otherwise
        """
        if current_clarity > previous_clarity:
            return False
        else:
            return True
    
    def start(self):
        """
        Starts the optimization process
        """
        previous_clarity = float('-inf')
        image = self.microscope_controller.get_image()
        current_clarity = self.clarity_metric(image)


        while not self.convergence_check(previous_clarity, current_clarity):
            print("image focus:", self.microscope_controller.get_current_focus())
            print("previous clarity:", previous_clarity)
            print("clarity:", current_clarity)
            gradient = self.gradient(current_clarity)
            print("gradient:", gradient)
            move = gradient*self.lr
            print("move:", move)
            self.microscope_controller.move(move)

            new_img = self.microscope_controller.get_image()
            previous_clarity = current_clarity
            current_clarity = self.clarity_metric(new_img)
            print()
        
        print("image focus:", self.microscope_controller.get_current_focus())
        print("previous clarity:", previous_clarity)
        print("clarity:", current_clarity)
        return self.microscope_controller.get_image().get_image_tensor()
    
def main():
    optimizer = Optimizer(
        clarity_metric=DummyClarityMetric,
        microscope_controller=DummyMicroscopeController(),
        lr=0.1
    )
    optimized_image = optimizer.start()
    cv2.imshow("optimized image", optimized_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
        
        




