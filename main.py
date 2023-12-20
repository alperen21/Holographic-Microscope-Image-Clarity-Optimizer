from optimization.Optimizer import Optimizer
from clarity.Clarity import LaplacianClarityMetric, ConvolutionalClarityMetric
from microscope.Microscope import DummyCropMicroscopeController

def main():
    laplacian = LaplacianClarityMetric("yolov8n.pt")
    optimizer = Optimizer(
        clarity_metric= laplacian,
        microscope_controller=DummyCropMicroscopeController(),
        lr=1
    )
    optimized_image = optimizer.start()
    print("optimized image clarity:", laplacian(optimized_image))

if __name__ == "__main__":
    main()