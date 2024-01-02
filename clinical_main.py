from microscope.Microscope import DummyClinicalMicroscopeController

def main():
    controller = DummyClinicalMicroscopeController()
    controller.move(2)
if __name__ == "__main__":
    main()