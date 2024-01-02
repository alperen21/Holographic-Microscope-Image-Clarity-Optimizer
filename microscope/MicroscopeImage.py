import cv2
from torch import tensor

class MicroscopeImage:
    def __init__(self, img_url=None) -> None:
        """
        MicroscopeImage is a wrapper class for an image tensor

        :img_url: url of the image
        """
        if img_url is not None:
            self.image = cv2.imread(img_url)
    
    def get_image_tensor(self) -> tensor:
        """
        Returns the image tensor

        :returns: image tensor 
        :rtype: tensor
        """
        return self.image
    
    def set_image_tensor(self, image : tensor) -> None:
        """
        Sets the image tensor

        :image: image tensor
        :returns: None
        """
        self.image = image

class DummyMicroscopeImage(MicroscopeImage):
    def __init__(self, img_focus, img_url) -> None:
        """
        DummyMicroscopeImage is a wrapper class for an image tensor and a focus plane for dummy images
        """
        super().__init__(img_url)
        self.img_focus = img_focus
    
    def get_focus(self) -> int:
        """
        Returns the focus plane

        :returns: focus plane
        :rtype: int
        """
        return self.img_focus
        