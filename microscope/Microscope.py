from abc import ABC, abstractmethod
import os 
import random
import cv2
from microscope.MicroscopeImage import DummyMicroscopeImage, MicroscopeImage
import math
import numpy as np
# import requests

class MicroscopeController(ABC):
    """
    Abstract class for controlling the microscope
    """
    def __init__(self, discrete_move = True, step_size = 0.5) -> None:
        self.discretize_move = discrete_move
        self.step_size = step_size

    @abstractmethod
    def get_image(self) -> MicroscopeImage:
        """
        Returns the current image
        
        :returns: MicroscopeImage
        :rtype: MicroscopeImage
        """
        pass

    @abstractmethod
    def move(self, move_amount : int) -> None:
        """
        Moves the microscope by move_amount, positive values moves it up and negative values moves it down

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: None
        """
        pass

    @abstractmethod
    def is_move_legal(self, move_amount : float) -> bool:
        """
        Checks if the move is legal given the current position of the actuator and the prospective move amount

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: bool
        """
        pass

    def discretize_move(self, move_amount : float) -> float:
        """
        Sets the move amount to the nearest multiple of the step size

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: float
        """
        num_steps = math.ceil(move_amount / self.step_size)
        return num_steps * self.step_size

class DummyMicroscopeController(MicroscopeController):
    def __init__(self, token="haydarpasa", folder="dummy_images") -> None:
        super().__init__()
        self.token = token
        self.folder = folder
        self.image_focuses = [-3.0, -2.5, -2.0, -1.0, -1.5, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.image_idx = random.randint(1, len(self.image_focuses)-2) #important this should not start with either of the extremes
        self.image = DummyMicroscopeImage(
            img_focus=self.image_focuses[self.image_idx],
            img_url=self.get_image_path()
        )
    
    def get_current_focus(self) -> int:
        """
        Returns the current focus plane

        :returns: current focus plane
        :rtype: int
        """
        return self.image_focuses[self.image_idx]
    
    def get_image_path(self) -> str:
        """
        Returns the path to the current image

        :returns: path to the current image
        """
        if self.image_focuses[self.image_idx] < 0.0:
            return os.path.join(self.folder, f"{self.token}{self.image_focuses[self.image_idx]}.jpg")
        else:
            return os.path.join(self.folder, f"{self.token}+{self.image_focuses[self.image_idx]}.jpg")
    
    def move(self, move_amount : float) -> int:
        """
        Discretizes the move and moves the dummy actuator by the move amount

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: None
        """
        move_amount = super().discretize_move(move_amount)
        if self.is_move_legal(move_amount):
            new_focus = self.image_focuses[self.image_idx] + move_amount
            self.image_idx = self.image_focuses.index(new_focus)
            self.image = DummyMicroscopeImage(
                img_focus=self.image_focuses[self.image_idx],
                img_url=self.get_image_path()
            )
        else:
            raise Exception("Move is not legal")
    
    def is_move_legal(self, move_amount : float) -> bool:
        """
        Checks if the move is legal given the current position of the actuator and the prospective move amount

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: bool
        """
        if self.image_idx + move_amount < 0 or self.image_idx + move_amount >= len(self.image_focuses):
            return False
        else:
            return True
    
    def get_image(self):
        return self.image
    

class DummyCropMicroscopeController(DummyMicroscopeController):
    """
    Dummy microscope controller that works with dummy crop images instead
    """
    def __init__(self) -> None:
        super().__init__(token="input", folder="dummy_crop_images")

class RestMicroscopeController(MicroscopeController):
    """
    Actual microscope controller that uses the REST API to communicate with the actuator
    """
    def __init__(self, discrete_move=True, step_size=0.5) -> None:
        super().__init__(discrete_move, step_size)
        self.authorization = ('ovizio', 'asdf123')
        self.api_url_obj = "http://localhost:9000/api/microscope/objectiveactuator"
        self.headers_obj = {"Content-Type": "application/json"}
        self.headers_acq = {"Accept": "image/png"}
        self.api_url_acq = "http://localhost:9000/api/microscope/cameras/0/frame"
        self.api_url_calibration = "http://localhost:9000/api/microscope/calibration"

        focus_plane = 0
        range = 5
        self.up_limit, self.low_limit = focus_plane + range, focus_plane - range

        self.height = self.get_height()
    
    def get_height(self) -> int:
        """
        Gets the current height of the actuator

        :returns: current height of the actuator
        :rtype: int
        """
        response = requests.get(self.api_url_calibration, 
                                headers=self.headers_obj,
                                auth=self.authorization)
        if response.status_code == 200:
            data = response.json()
            height = int(data.get('Height', 'Height data not found'))
            return height
        else:
            raise Exception("Could not get height")

    def get_image(self) -> MicroscopeImage:
        """
        Returns the current image
        
        :returns: MicroscopeImage
        :rtype: MicroscopeImage
        """
        pic = requests.get(
            self.api_url_obj,
            headers=self.headers_acq,
            auth=self.authorization
        )

        if pic.status_code == 200:
            microscopeImage = MicroscopeImage()
            nparr = np.frombuffer(pic.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            microscopeImage.set_image_tensor(img)
            return microscopeImage
        else:
            raise Exception("Could not get image")

    def move(self, move_amount : float) -> None:
        """
        Moves the microscope by move_amount, positive values moves it up and negative values moves it down

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: None
        """
        new_height = move_amount + self.height

        if self.is_move_legal(new_height):
            data = {
                "Position": new_height
            }
            pos = requests.put(self.api_url_obj, headers=self.headers_obj, auth=self.authorization, data=data)
            if pos.status_code == 200:
                self.height = new_height
            else:
                raise Exception("Could not move")
        else:
            raise Exception("Move is not legal")


    def is_move_legal(self, move_amount : float) -> bool:
        """
        Checks if the move is legal given the current position of the actuator and the prospective move amount

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: bool
        """
        return self.height + move_amount <= self.up_limit and self.height + move_amount >= self.low_limit


class DummyClinicalMicroscopeController(MicroscopeController):
    """
    Dummy microscope controller that works with clinical images instead
    """
    def __init__(self, start_token="534", end_token="_phase.png", reference_token="ref", folder="clinical_images") -> None:
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.folder = folder
        self.reference_token = reference_token


        self.image_focuses = [
            "+5",
            "+3",
            "ref",
            "-3",
            "-5"
        ]
        self.image_idx = random.randint(1, len(self.image_focuses)-2) #important this should not start with either of the extremes
        self.image = MicroscopeImage(
            img_url=self.get_image_path()
        )

    def get_image_path(self) -> str:
        """
        Returns the path to the current image

        :returns: path to the current image
        """
        return os.path.join(self.folder, f"{self.start_token}_{121-int(0 if self.image_focuses[self.image_idx] == self.reference_token else self.image_focuses[self.image_idx])}_{self.image_focuses[self.image_idx]}{self.end_token}")

    def get_image(self) -> MicroscopeImage:
        """
        Returns the current image
        
        :returns: MicroscopeImage
        :rtype: MicroscopeImage
        """
        return self.image
    
    def move(self, move_amount : float) -> None:
        """
        Moves the microscope by move_amount, positive values moves it up and negative values moves it down

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: None
        """
        move_amount = int(super().discretize_move(move_amount))

        focus = str(int(0 if self.image_focuses[self.image_idx] == self.reference_token else self.image_focuses[self.image_idx]) + move_amount)
        focuses = self.image_focuses.copy()
        focuses[focuses.index(self.reference_token)] = "0"
        focus = str(min([int(focus) for focus in focuses], key=lambda x: abs(x - int(focus))))

        if int(focus) > 0:
            focus = "+" + focus
        elif int(focus) == 0:
            focus = self.reference_token
        
        if self.is_move_legal(move_amount):
            new_focus = focus
            self.image_idx = self.image_focuses.index(new_focus)
            self.image = MicroscopeImage(
                img_url=self.get_image_path()
            )
        else:
            raise Exception("Move is not legal")

    def is_move_legal(self, move_amount : float) -> bool:
        """
        Checks if the move is legal given the current position of the actuator and the prospective move amount

        :param move_amount: amount to move the microscope
        :type move_amount: float
        :rtype: bool
        """
        return True
    
    def get_current_focus(self) -> int:
        """
        Returns the current focus plane

        :returns: current focus plane
        :rtype: int
        """
        return self.image_focuses[self.image_idx]

if __name__ == "__main__":
    controller = DummyMicroscopeController()

    for i in range(13):
        controller.image_idx = i
        print(controller.get_image_path())

    
    
    


