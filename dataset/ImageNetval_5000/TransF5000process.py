import os
import json
import shutil
from tqdm import tqdm

# Load the JSON file
with open('TransF5000_val.json', 'r') as json_file:
    data = json.load(json_file)

oripath = 'D:\data\ILSVRC2012_img_val'
# Get the root directory path from JSON
root_path = data["root"]

# Create output directories if they don't exist
if not os.path.exists(root_path):
    os.makedirs(root_path)

# Iterate through the samples in the JSON file and copy the images to the corresponding class directories
for sample in tqdm(data["samples"]):
    image_path, class_idx = sample
    class_name = data["classes"][class_idx]
    class_dir = os.path.join(class_name)

    root = "val5000"
    # Create class directory if it doesn't exist
    if not os.path.exists(class_dir):
        os.makedirs(os.path.join(root, class_dir), exist_ok=True)

    # Extract the relative path of the image within val directory
    relative_path = os.path.basename(image_path)
    # Construct the new image path in the /val5000/ directory
    new_image_path = os.path.join(root_path, class_dir, relative_path)

    original_image_path = os.path.join(oripath, class_dir, relative_path)
    # Copy the image from the original val directory to the corresponding /val5000/ directory
    shutil.copy(original_image_path, new_image_path)

print("Image dataset generation completed.")
