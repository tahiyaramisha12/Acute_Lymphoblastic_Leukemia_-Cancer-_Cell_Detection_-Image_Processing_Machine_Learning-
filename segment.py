import cv2
import os
import numpy as np

input_dir = 'preprocessed_dataset'
output_dir = 'segmented_dataset'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

categories = {
    'cancer': ['Benign', 'Early Malignant', 'Pre Malignant', 'Pro Malignant'],
    'normal': []
}

def segment_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing

for category, subcategories in categories.items():
    if subcategories:
        for subcategory in subcategories:

            input_subcategory_dir = os.path.join(input_dir, category, subcategory)
            output_subcategory_dir = os.path.join(output_dir, category, subcategory)

            if not os.path.exists(output_subcategory_dir):
                os.makedirs(output_subcategory_dir)

            if os.path.exists(input_subcategory_dir):
                images = os.listdir(input_subcategory_dir)

                for image_name in images:
                    image_path = os.path.join(input_subcategory_dir, image_name)

                    segmented_image = segment_image(image_path)

                    output_image_path = os.path.join(output_subcategory_dir, image_name)
                    cv2.imwrite(output_image_path, segmented_image)

    else:
        input_category_dir = os.path.join(input_dir, category)
        output_category_dir = os.path.join(output_dir, category)

        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)

        if os.path.exists(input_category_dir):
            images = os.listdir(input_category_dir)

            for image_name in images:
                image_path = os.path.join(input_category_dir, image_name)

                segmented_image = segment_image(image_path)

                output_image_path = os.path.join(output_category_dir, image_name)
                cv2.imwrite(output_image_path, segmented_image)


print("Image segmentation complete.")