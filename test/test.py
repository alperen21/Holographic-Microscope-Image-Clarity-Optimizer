from Clarity import brenner, Laplacian, SMD, SMD2, variance, energy, harmonic_mean, geometric_mean, arithmetic_mean, Vollath, Tenengrad, entropy
import cv2
import os
import time

def test(metric):
    time_start = time.time()
    for i in [0.5, 1.0, 2.0, 2.5, 3.0]:
        img = cv2.imread(os.path.join("dummy_images",f"haydarpasa-{i}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        print(i, "->", metric(img))
    time_end = time.time()
    print("time elapsed:", time_end-time_start, "seconds")

def main():
    for metric in [brenner, Laplacian, SMD, SMD2, variance, energy, Vollath, Tenengrad, entropy, harmonic_mean, geometric_mean, arithmetic_mean]:
        try:
            print(metric.__name__)
            test(metric)
            print()
        except Exception as e:
            print("an exception occured", e)



if __name__ == "__main__":
    main()