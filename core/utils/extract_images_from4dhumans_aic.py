import webdataset as wds
import os
import cv2
import sys
from tqdm import tqdm

def extract_images(start_index, end_index, tar_url_template, output_dir):
    start = int(start_index)
    end = int(end_index)

    # Generate list of tar file paths
    urls = [
        tar_url_template.format(i=idx)
        for idx in range(start, end + 1)
    ]

    os.makedirs(output_dir, exist_ok=True)

    dataset = (
        wds.WebDataset(urls)
        .decode("rgb8")
        .rename(jpg="jpg;jpeg;png")
    )

    # Iterate over the dataset to save images
    for i, sample in tqdm(enumerate(dataset)):
        try:
            # Extract key and image data
            key = sample["__key__"]
            image_data = sample["jpg"]

            # Construct the full output path for the image
            base_dir = os.path.dirname(key).replace("aic-train-vitpose", "images")
            filename = os.path.basename(key) + ".jpg"
            image_path_dir = os.path.join(output_dir, base_dir)

            # Create the output directory if it doesn't exist
            os.makedirs(image_path_dir, exist_ok=True)
            
            image_path = os.path.join(image_path_dir, filename)

            # Save the image using OpenCV
            cv2.imwrite(image_path, cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

if __name__ == "__main__":
    # Parse command-line arguments for start and end indices
    start_index = sys.argv[1]
    end_index = sys.argv[2]

    # Define the URL template and output directory
    # tar_url_template = "data/4DHumans/insta-train-vitpose-replicate/{0..{1}}.tar" #Download from 4D-Humans website
    # output_dir = "data/training-images/insta"

    tar_url_template = "/ssd/common/datasets/pose-analysis/aic/{i:06d}.tar" #Download from 4D-Humans website
    output_dir = "/ssd/common/datasets/pose-analysis/aic"

    # Run the extraction function
    extract_images(start_index, end_index, tar_url_template, output_dir)
