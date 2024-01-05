# Holographic Microscope Image Clarity Optimizer
This project aims to automatize focusing process for holographic microscopes by optimizing clarity metrics.
The process uses 3 phases:
* **Cropping**: A YOLO image recognition transformer is used to detect cells and crop cell images based on detection results. If no cell is detected and thus no crops are generated, the process will be repeated until a cell is found
* **Clarity metric**: Cropped images will be used as an input of the selected clarity metric and an average final clarity metric will be generated and normalized
* **Gradient ascent**: The microscope controller will move the actuator very slightly and contrast the current clarity with the previous clarity to find the gradient.
  If the current gradient is worse, the optimizer will reposition the actuator the opposite direction with the same magnitude from the original position.
  If the clarity is worse once again, it will conclude that the clarity has converged and the global maxima has been found. If not the algorithm will continue running, in each case the actuator will be repositioned to the original position.
  If convergence check has failed, the position of the actuator will be changed according to the result of the multiplication of the gradient and the learning rate.

## Submodules
### Clarity Metrics
Clarity metrics are included under clarity submodule. Mathematical, stand-alone clarity metrics included are:
* Laplacian
* Brenner
* SMD
* SMD2
* Variance
* Energy
* Vollath
* Inverse Entropy
* Entropy
* Tenengrad

There is also 3 mean clarity metrics that use all mathematical stand-alone clarity metrics:
* Arithmetic mean
* Geometric mean
* Harmonic mean

This submodule also includes Keras Clarity Metric which uses a model that is trained using focused and out-of-focus images of cells and the first element of the last layer's activation value is scaled and used as a clarity metric

### Cropping
Crops cells out of the image. Cells that are not fully captured by the microscope will be discarded

### Microscope
Contains microscope controllers including dummy micrscope controllers for demonstration purposes and a Rest Api microscope controller to communicate with the actuator of the microscope using http requests

### Optimization
The optimizers utilize certain algorithms to find the optimal distance of the actuator.

#### GradOptimizer
Grad optimizer utilizes gradient ascent algorithm to optimize the distance

#### RapidOptimizer
Rapid optimizer uses inference from ResNet18 to predict how far away the actuator is from the optimal focal point and moves the actuator appropriately. This algorithm is especially useful when quick autofocus is required.

### Configuration
Contains configurations:
* lr: learning rate
* gradient step: how many milimeters the actuator should move for calculating the gradient
* crop: crop the cell images and calculate the average clarity if set to True. If set to False, it treats the whole image as one crop and calculates the clarity only once
* model: path to the model weights which will be used for cropping the cell images

## Usage
Modify (if needed) the configuration file found under configuration submodule
Instantiate the desired clarity metric class using the desired model for cropping the cell images
```
from clarity.Clarity import InverseEntropyClarityMetric

clarityMetric = InverseEntropyClarityMetric()
```

Instantiate the optimizer with a desired microscope controller
```
from microscope.Microscope import DummyClinicalMicroscopeController
from optimization.Optimizer import Optimizer

optimizer = Optimizer(
    clarity_metric= clarityMetric,
    microscope_controller=DummyClinicalMicroscopeController(),
)
```

Start the optimizer
```
optimizer.start()
```

  
