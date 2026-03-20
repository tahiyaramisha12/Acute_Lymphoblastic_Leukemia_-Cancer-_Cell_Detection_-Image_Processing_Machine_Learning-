
import cv2
import os
import numpy as np

input_dir = 'dataset'
output_dir = 'preprocessed_dataset'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

categories = {
    'cancer': ['Benign', 'Early Malignant', 'Pre Malignant', 'Pro Malignant'],
    'normal': []
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    equalized = cv2.equalizeHist(blurred)
    
    normalized = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX)
    
    edges = cv2.Canny(normalized, 100, 200)
    
    return normalized

for category, subcategories in categories.items():
    if subcategories:
        for subcategory in subcategories:
            output_subcategory_dir = os.path.join(output_dir, category, subcategory)
            if not os.path.exists(output_subcategory_dir):
                os.makedirs(output_subcategory_dir)
            
            input_subcategory_dir = os.path.join(input_dir, category, subcategory)
            if os.path.exists(input_subcategory_dir):
                images = os.listdir(input_subcategory_dir)
                
                for image_name in images:
                    image_path = os.path.join(input_subcategory_dir, image_name)
                    processed_image = preprocess_image(image_path)
                    output_image_path = os.path.join(output_subcategory_dir, image_name)
                    cv2.imwrite(output_image_path, processed_image)
    else:
        output_category_dir = os.path.join(output_dir, category)
        if not os.path.exists(output_category_dir):
            os.makedirs(output_category_dir)
            
        input_category_dir = os.path.join(input_dir, category)
        if os.path.exists(input_category_dir):
            images = os.listdir(input_category_dir)
            
            for image_name in images:
                image_path = os.path.join(input_category_dir, image_name)
                processed_image = preprocess_image(image_path)
                output_image_path = os.path.join(output_category_dir, image_name)
                cv2.imwrite(output_image_path, processed_image)

print("Image preprocessing complete.")
