import os
import cv2
import numpy as np

def extract_bounding_box(image, bbox):
    """
    Extract the bounding box area from an image based on the given coordinates.

    Args:
    image (numpy.ndarray): Input image.
    bbox (list): Bounding box coordinates in the format [x, y, w, h].

    Returns:
    numpy.ndarray: Extracted bounding box area.
    """
    x, y, w, h = bbox

    # Adjust bounding box coordinates and apply padding if necessary
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0

    return image[y:y+h, x:x+w]

def resize_with_padding(image, target_size):
    """
    Resize the image with padding to the target size.

    Args:
    image (numpy.ndarray): Input image.
    target_size (tuple): Target size (width, height).

    Returns:
    numpy.ndarray: Resized image with padding.
    """
    width, height = target_size
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Calculate resizing dimensions
    if aspect_ratio > 1:
        new_w = width
        new_h = int(width / aspect_ratio)
    else:
        new_h = height
        new_w = int(height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a canvas with padding
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    return canvas

def process_images(image_folder, annotation_file, output_folder,output_folder2, target_size=(127, 127), bbox_size=(63, 63)):
    """
    Process images in the specified folder using the bounding box information provided in the annotation file.

    Args:
    image_folder (str): Path to the folder containing images.
    annotation_file (str): Path to the text file containing bounding box annotations.
    output_folder (str): Path to the folder where extracted images will be saved.
    target_size (tuple): Target size for the final output area (width, height).
    bbox_size (tuple): Size for the bounding box area (width, height).
    """
    with open(annotation_file, 'r') as file:
        annotations = file.readlines()

    for idx, annotation in enumerate(annotations):
        image_name = str(idx) + '.jpg'
        bbox = list(map(int, annotation.strip().split(' ')))
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        # Extract the bounding box area
        bounding_box_area = extract_bounding_box(image, bbox)

        # Resize the bounding box area to target size
        resized_bbox = cv2.resize(bounding_box_area, bbox_size)

        # Calculate resizing ratio for original image
        ratio = max(bounding_box_area.shape[0], bounding_box_area.shape[1]) / max(bbox_size)

        # Resize the original image with padding
        resized_image = resize_with_padding(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)))

        # Extract the 127x127 area with bounding box at the center
        center_x = bbox[0] / ratio + bbox[2] / 2
        center_y = bbox[1] / ratio + bbox[3] / 2
        top_left_x = max(int(center_x - target_size[0] / 2), 0)
        top_left_y = max(int(center_y - target_size[1] / 2), 0)
        bottom_right_x = min(int(center_x + target_size[0] / 2), resized_image.shape[1])
        bottom_right_y = min(int(center_y + target_size[1] / 2), resized_image.shape[0])

        # Crop the region
        extracted_area = resized_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # If extracted area is smaller than target size, pad it with mirror edges
        if extracted_area.shape[0] < target_size[0] or extracted_area.shape[1] < target_size[1]:
            pad_top = max(0, (target_size[0] - extracted_area.shape[0]) // 2)
            pad_bottom = max(0, target_size[0] - extracted_area.shape[0] - pad_top)
            pad_left = max(0, (target_size[1] - extracted_area.shape[1]) // 2)
            pad_right = max(0, target_size[1] - extracted_area.shape[1] - pad_left)
            extracted_area = cv2.copyMakeBorder(extracted_area, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

        # Save the extracted area
        output_path1 = os.path.join(output_folder, 'output_' + image_name)
        output_path2 = os.path.join(output_folder2, 'output_' + image_name)
        cv2.imwrite(output_path1, extracted_area)
        target = extracted_area[32:95, 32:95]
        cv2.imwrite(output_path2,target)

        print(idx)


# Example usage:
image_folder = '/home/ran/HDD/Data/Benchmark/OTB/Data/Biker/train/'
annotation_file = '/home/ran/HDD/Data/Benchmark/OTB/Data/Biker/train/coordinates.txt'
output_folder = '/home/ran/HDD/Data/Benchmark/OTB/Data/Biker/train/output_images/'
output_folder2 = '/home/ran/HDD/Data/Benchmark/OTB/Data/Biker/train/targets/'
process_images(image_folder, annotation_file, output_folder,output_folder2)
