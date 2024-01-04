from microscope.Microscope import DummyMicroscopeController, MicroscopeController, MicroscopeImage
from clarity.Clarity import DummyClarityMetric, LaplacianClarityMetric, LinearClarityMetric
import cv2
from configurations.config import config


convergence_trials = 0
class Optimizer:
    def __init__(self, clarity_metric : LinearClarityMetric, microscope_controller : MicroscopeController, lr : float =config["lr"]) -> None:
        """
        :param clarity_metric: function that takes in an image and returns a clarity score
        :param microscope_controller: object that controls the microscope
        :param lr: learning rate
        """
        self.lr = lr
        self.clarity_metric = clarity_metric
        self.microscope_controller = microscope_controller

    def forward(self) -> float:
        """
        Returns the clarity of the current image

        :returns: clarity of the current image
        :rtype: float
        """
        image = self.microscope_controller.get_image()
        clarity = self.clarity_metric(image)
        return clarity

        
    def gradient(self, previous_clarity : float, gradient_step : float =config["gradient step"]) -> float:
        """
        Returns the gradient of the clarity metric with respect to the move amount

        :returns: gradient of the clarity metric with respect to the move amount
        :rtype: float
        """
        self.microscope_controller.move(gradient_step)
        image = self.microscope_controller.get_image()
        new_clarity = self.clarity_metric(image)
        print(previous_clarity)
        print(new_clarity)
        grad = new_clarity - previous_clarity
        self.microscope_controller.move((-1)*gradient_step)

        
        if new_clarity < previous_clarity:
            self.microscope_controller.move((-1)*gradient_step)
            image = self.microscope_controller.get_image()
            new_clarity_2 = self.clarity_metric(image)
            self.microscope_controller.move(gradient_step)


            if new_clarity_2 < previous_clarity:
                return 0
        
        return grad
    
    def convergence_check(self, previous_clarity : float, current_clarity : float) -> bool:
        """
        Returns True if the optimizer has converged, False otherwise

        :param previous_clarity: clarity of the previous image
        :param current_clarity: clarity of the current image
        
        :returns: True if the optimizer has converged, False otherwise
        :rtype: bool
        """
        global convergence_trials
        convergence_trials += 1

        if convergence_trials > config["max convergence trials"]:
            return True

        if current_clarity > previous_clarity:
            return False
        else:
            return True
    
    def start(self) -> MicroscopeImage:
        """
        Starts the optimization process

        :returns: optimized image
        :rtype: MicroscopeImage
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
        return self.microscope_controller.get_image()
    
def main():
    laplacianClarityMetric = LaplacianClarityMetric(model_weights="yolov8n.pt")
    optimizer = Optimizer(
        clarity_metric=laplacianClarityMetric,
        microscope_controller=DummyMicroscopeController(),
        lr=1
    )
    optimized_image = optimizer.start()
    cv2.imshow("optimized image", optimized_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
        
        




