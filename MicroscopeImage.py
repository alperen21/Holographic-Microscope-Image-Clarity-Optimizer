import cv2

class MicroscopeImage:
    def __init__(self, img_url=None) -> None:
        if img_url is not None:
            self.image = cv2.imread(img_url)
    
    def get_image_tensor(self):
        return self.image
    
    def set_image_tensor(self, image):
        self.image = image

class DummyMicroscopeImage(MicroscopeImage):
    def __init__(self, img_focus, img_url) -> None:
        super().__init__(img_url)
        self.img_focus = img_focus
    
    def get_focus(self):
        return self.img_focus
        