from clarity.Clarity import ConvolutionalClarityMetric, LaplacianClarityMetric, BrennerClarityMetric, SMDClarityMetric, SMD2ClarityMetric, VarianceClarityMetric, EnergyClarityMetric, VollathClarityMetric, EntropyClarityMetric,TenengradClarityMetric
from microscope.Microscope import MicroscopeImage
import os 

def get_images(data_dir, image_type):
    dir_path = f'data2/{data_dir}'

    # Listing all files in the specified directory
    files = os.listdir(dir_path)

    # Filtering out only images
    image_files = [file for file in files if image_type in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]




    return image_files


def test_metric(clarityMetric):
    images = {
        "-5":[],
        "-3":[],
        "ref":[],
        "+3":[],
        "+5":[],
    }

    results = {
        "-5":0,
        "-3":0,
        "ref":0,
        "+3":0,
        "+5":0,
    }
    keys = images.keys()
    for key in keys:
        images[key] = get_images(key, "phase")
    
    
    # print(get_images(-3, "amplitude"))
    #clarityMetric = LaplacianClarityMetric("24_10_23_yolov8x_no_aug_iou_0.7.pt")

    for key in keys:
        count = 0
        for image in images[key]:
            
            
            image_object = MicroscopeImage(f"data2/{key}/{image}")
            print(image_object)

            clarity = clarityMetric.get_clarity(image_object)
            results[key] += float(clarity)* 1000
            count += 1
            print(key,count)
        results[key] /= count
    
    with open(f"results_{clarityMetric.__class__.__name__}.txt", "w") as f:
        for key in results.keys():
            f.write(f"{key} {results[key]}\n")

def main():
    model = "24_10_23_yolov8x_no_aug_iou_0.7.pt"

    metrics = [
        # LaplacianClarityMetric(model),
        # BrennerClarityMetric(model),
        # ConvolutionalClarityMetric(model),
        # SMDClarityMetric(model),
        # SMD2ClarityMetric(model),
        # VarianceClarityMetric(model),
        # EnergyClarityMetric(model),
        # VollathClarityMetric(model),
        EntropyClarityMetric(model),
        TenengradClarityMetric(model)

    ]

    for metric in metrics:
        try:
            test_metric(metric)
        except Exception as e:
            with open("exceptions.txt", "a") as f:
                f.write(f"{metric.__class__.__name__} {e}\n")
                raise e




if __name__ == "__main__":
    main()
