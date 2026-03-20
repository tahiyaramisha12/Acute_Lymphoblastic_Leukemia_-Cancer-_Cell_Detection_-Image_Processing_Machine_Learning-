import os
import cv2
from pathlib import Path

def check_dataset():
    """Check if dataset is properly structured"""
    base_path = 'dataset'
    
    print("Checking dataset structure...")
    
    normal_path = os.path.join(base_path, 'normal')
    if os.path.exists(normal_path):
        normal_images = [f for f in os.listdir(normal_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Normal images: {len(normal_images)}")
        
        for img in normal_images[:5]: 
            img_path = os.path.join(normal_path, img)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print(f"  Warning: Cannot read {img}")
    else:
        print(f"ERROR: {normal_path} not found!")
    
    cancer_types = ['Benign', 'Early Malignant', 'Pre Malignant', 'Pro Malignant']
    cancer_path = os.path.join(base_path, 'cancer')
    
    for cancer_type in cancer_types:
        type_path = os.path.join(cancer_path, cancer_type)
        if os.path.exists(type_path):
            images = [f for f in os.listdir(type_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"{cancer_type}: {len(images)} images")
            
            for img in images[:5]:
                img_path = os.path.join(type_path, img)
                test_img = cv2.imread(img_path)
                if test_img is None:
                    print(f"  Warning: Cannot read {img}")
        else:
            print(f"ERROR: {type_path} not found!")

if __name__ == "__main__":
    check_dataset()